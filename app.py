"""Pocket Smart AI - Your Smart Budget & Recommendation Assistant."""
import json
import os
import sqlite3
import time
from datetime import date
from functools import wraps

import markdown as md
from dotenv import load_dotenv
from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from markupsafe import Markup, escape
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "pocket.db")
ALLOWED_IMG = {"png", "jpg", "jpeg", "webp"}
CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Health", "Entertainment", "Other"]

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
    UPLOAD_FOLDER=os.path.join(BASE, "static", "uploads"),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ---------------------------------------------------------------- database
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        monthly_budget REAL DEFAULT 20000);
    CREATE TABLE IF NOT EXISTS expenses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        spent_on TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS plans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        kind TEXT NOT NULL,
        budget REAL NOT NULL,
        details TEXT,
        result TEXT NOT NULL,
        image TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    conn.commit()
    conn.close()


# ----------------------------------------------------------------- helpers
def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


def render_md(text):
    """Escape first, then convert markdown, so model output can't inject HTML."""
    return Markup(md.markdown(str(escape(text)), extensions=["tables", "nl2br"]))


@app.template_filter("inr")
def inr(value):
    return "₹{:,.0f}".format(value or 0)


@app.template_filter("md")
def md_filter(text):
    return render_md(text)


@app.template_filter("fromjson")
def fromjson(s):
    try:
        return json.loads(s or "{}")
    except ValueError:
        return {}


def save_upload(file):
    if not file or not file.filename:
        return None
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMG:
        flash("Reference image ignored: use PNG, JPG or WEBP.", "error")
        return None
    name = f"{session['user_id']}_{int(time.time())}_{secure_filename(file.filename)}"
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], name))
    return name


def month_summary(uid):
    month = date.today().strftime("%Y-%m")
    rows = db().execute(
        "SELECT category, SUM(amount) t FROM expenses WHERE user_id=? AND spent_on LIKE ? GROUP BY category",
        (uid, month + "%")).fetchall()
    by_cat = {r["category"]: r["t"] for r in rows}
    budget = db().execute("SELECT monthly_budget FROM users WHERE id=?", (uid,)).fetchone()[0]
    return by_cat, sum(by_cat.values()), budget


# --------------------------------------------------------------------- LLM
def ask_llm(system, prompt):
    """Return model text, or None when no key is set / the call fails."""
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None, timeout=40)
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.6, max_tokens=1000)
        return resp.choices[0].message.content
    except Exception as exc:  # network, auth, quota...
        app.logger.error("LLM call failed: %s", exc)
        return None


SYSTEM = ("You are Pocket Smart AI, a practical budgeting and shopping-recommendation "
          "assistant for users in India. Use Indian rupees (₹). Be specific, realistic and "
          "concise. Answer in markdown: a short summary, a budget split table "
          "(item | share | amount), 3-5 concrete recommendations, and 2 money-saving tips. "
          "Never exceed the user's budget.")

PLANNERS = {
    "home": dict(title="Home planner", route="/home-planner", template="home_planner.html",
                 split=[("Furniture", 40), ("Decor & soft furnishings", 20), ("Lighting", 15),
                        ("Storage & organisers", 15), ("Contingency", 10)]),
    "jewelry": dict(title="Jewelry planner", route="/jewelry-planner", template="jewelry_planner.html",
                    split=[("Main piece", 60), ("Matching accessories", 20),
                           ("Making / hallmarking charges", 12), ("Insurance & box", 8)]),
    "party": dict(title="Party planner", route="/party-planner", template="party_planner.html",
                  split=[("Venue", 30), ("Food & drinks", 35), ("Decor", 15),
                         ("Entertainment", 10), ("Contingency", 10)]),
}


def fallback_plan(kind, budget, details):
    rows = "\n".join(f"| {n} | {p}% | ₹{budget * p / 100:,.0f} |" for n, p in PLANNERS[kind]["split"])
    notes = ", ".join(f"{k}: {v}" for k, v in details.items() if k != "budget" and v)
    return (f"### {PLANNERS[kind]['title']} - ₹{budget:,.0f}\n"
            f"Built from your inputs ({notes or 'no extra preferences'}).\n\n"
            f"| Item | Share | Amount |\n|---|---|---|\n{rows}\n\n"
            "**Tips**\n\n- Compare at least three vendors or stores before paying.\n"
            "- Keep the contingency untouched until the last week.\n\n"
            "*Offline planner in use. Add `LLM_API_KEY` in `.env` for tailored AI recommendations.*")


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        if not name or "@" not in email or len(pw) < 6:
            flash("Enter your name, a valid email and a password of 6+ characters.", "error")
        elif db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            flash("That email is already registered. Log in instead.", "error")
        else:
            db().execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                         (name, email, generate_password_hash(pw)))
            db().commit()
            flash("Account created. Log in to start.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session["user_id"], session["name"] = user["id"], user["name"]
            return redirect(url_for("dashboard"))
        flash("Email or password is incorrect.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    uid = session["user_id"]
    by_cat, spent, budget = month_summary(uid)
    expenses = db().execute(
        "SELECT * FROM expenses WHERE user_id=? ORDER BY spent_on DESC, id DESC LIMIT 15", (uid,)).fetchall()
    return render_template("dashboard.html", by_cat=by_cat, spent=spent, budget=budget,
                           expenses=expenses, categories=CATEGORIES, today=date.today().isoformat())


@app.route("/expense/add", methods=["POST"])
@login_required
def add_expense():
    try:
        amount = float(request.form.get("amount", ""))
        assert amount > 0
    except (ValueError, AssertionError):
        flash("Enter an amount greater than zero.", "error")
        return redirect(url_for("dashboard"))
    cat = request.form.get("category")
    db().execute("INSERT INTO expenses(user_id,title,category,amount,spent_on) VALUES(?,?,?,?,?)",
                 (session["user_id"], request.form.get("title", "").strip() or cat,
                  cat if cat in CATEGORIES else "Other", amount,
                  request.form.get("spent_on") or date.today().isoformat()))
    db().commit()
    flash("Expense added.", "success")
    return redirect(url_for("dashboard"))


@app.route("/expense/<int:eid>/delete", methods=["POST"])
@login_required
def delete_expense(eid):
    db().execute("DELETE FROM expenses WHERE id=? AND user_id=?", (eid, session["user_id"]))
    db().commit()
    flash("Expense deleted.", "success")
    return redirect(url_for("dashboard"))


@app.route("/budget", methods=["POST"])
@login_required
def set_budget():
    try:
        value = float(request.form.get("budget", ""))
        assert value > 0
        db().execute("UPDATE users SET monthly_budget=? WHERE id=?", (value, session["user_id"]))
        db().commit()
        flash("Monthly budget updated.", "success")
    except (ValueError, AssertionError):
        flash("Enter a budget greater than zero.", "error")
    return redirect(url_for("dashboard"))


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(html="<p>Type a question first.</p>")
    by_cat, spent, budget = month_summary(session["user_id"])
    context = (f"This month: budget ₹{budget:,.0f}, spent ₹{spent:,.0f}, "
               f"by category {json.dumps(by_cat)}.\nQuestion: {question}")
    answer = ask_llm(SYSTEM, context)
    if not answer:
        left = budget - spent
        top = max(by_cat, key=by_cat.get) if by_cat else None
        answer = (f"You have **₹{left:,.0f}** left of ₹{budget:,.0f} "
                  f"({spent / budget * 100:.0f}% used).\n\n"
                  + (f"Your biggest category is **{top}** (₹{by_cat[top]:,.0f}). Trim it first.\n\n" if top else "")
                  + "*Offline mode: add `LLM_API_KEY` in `.env` for full AI answers.*")
    return jsonify(html=str(render_md(answer)))


def make_planner(kind):
    cfg = PLANNERS[kind]

    @login_required
    def view():
        result, image, form = None, None, {}
        if request.method == "POST":
            form = request.form.to_dict()
            try:
                budget = float(form.get("budget", ""))
                assert budget > 0
            except (ValueError, AssertionError):
                flash("Enter a budget greater than zero.", "error")
                return render_template(cfg["template"], result=None, image=None, form=form)
            image = save_upload(request.files.get("reference"))
            details = {k: v.strip() for k, v in form.items() if k != "budget" and v.strip()}
            prompt = (f"Create a {cfg['title'].lower()} for a total budget of ₹{budget:,.0f}.\n"
                      + "\n".join(f"- {k}: {v}" for k, v in details.items()))
            text = ask_llm(SYSTEM, prompt) or fallback_plan(kind, budget, details)
            db().execute("INSERT INTO plans(user_id,kind,budget,details,result,image) VALUES(?,?,?,?,?,?)",
                         (session["user_id"], kind, budget, json.dumps(details), text, image))
            db().commit()
            result = render_md(text)
        return render_template(cfg["template"], result=result, image=image, form=form)

    app.add_url_rule(cfg["route"], endpoint=f"{kind}_planner", view_func=view, methods=["GET", "POST"])


for _kind in PLANNERS:
    make_planner(_kind)


@app.route("/history")
@login_required
def history():
    plans = db().execute("SELECT * FROM plans WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("history.html", plans=plans, planners=PLANNERS)


@app.route("/history/<int:pid>/delete", methods=["POST"])
@login_required
def delete_plan(pid):
    db().execute("DELETE FROM plans WHERE id=? AND user_id=?", (pid, session["user_id"]))
    db().commit()
    flash("Plan deleted.", "success")
    return redirect(url_for("history"))


init_db()

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "1") == "1", port=5000)
