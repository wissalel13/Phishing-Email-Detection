"""Page web de l'interface, intégrée au code : aucun fichier statique à déployer."""

INDEX_HTML = r'''<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhishGuard AI</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --text: #1c2330; --muted: #5d6879; --line: #d9dee6;
    --accent: #2457d6; --bad: #b3261e; --bad-bg: #fdecea; --ok: #1e7a3c; --ok-bg: #e8f5ec; --warn: #8a5a00; --warn-bg: #fff4d6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12161d; --card: #1b212b; --text: #e6e9ef; --muted: #98a2b3; --line: #2d3644;
      --accent: #7ba3ff; --bad: #ff8b84; --bad-bg: #3a1f1d; --ok: #7fd49a; --ok-bg: #1b3325; --warn: #f3c969; --warn-bg: #3a2f14;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  label { display: block; font-weight: 600; margin: 0 0 6px; }
  input, textarea { width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); color: var(--text); font: inherit; }
  textarea { min-height: 170px; resize: vertical; }
  .row { margin-bottom: 12px; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button { font: inherit; padding: 9px 16px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--text); cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button:disabled { opacity: .6; cursor: progress; }
  .hint { color: var(--muted); font-size: .875rem; }
  #status { font-size: .875rem; color: var(--muted); }
  #status.bad { color: var(--bad); }
  .verdict { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; }
  .verdict.phishing { background: var(--bad-bg); color: var(--bad); }
  .verdict.legitime { background: var(--ok-bg); color: var(--ok); }
  .bar { height: 10px; border-radius: 999px; background: var(--line); overflow: hidden; margin: 8px 0 4px; }
  .bar > span { display: block; height: 100%; background: var(--bad); }
  .banner { background: var(--warn-bg); color: var(--warn); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 12px 0 0; }
  dt { color: var(--muted); }
  dd { margin: 0; overflow-wrap: anywhere; }
  .hidden { display: none; }
  footer { color: var(--muted); font-size: .8125rem; }
</style>
</head>
<body>
<main>
  <h1>PhishGuard AI</h1>
  <p class="sub">Analyse d'un email : modèle ML + LLM local (Ollama). Le texte reste sur cette machine.</p>

  <form class="card" id="form">
    <div class="row">
      <label for="subject">Objet (facultatif)</label>
      <input id="subject" maxlength="1000" autocomplete="off">
    </div>
    <div class="row">
      <label for="body">Contenu de l'email</label>
      <textarea id="body" required placeholder="Collez ici le texte de l'email…"></textarea>
    </div>
    <div class="actions">
      <button class="primary" id="go" type="submit">Analyser</button>
      <button type="button" id="ex1">Exemple suspect</button>
      <button type="button" id="ex2">Exemple légitime</button>
      <span id="status" role="status" aria-live="polite"></span>
    </div>
    <p class="hint">Une analyse prend en général 10 à 15 secondes (le LLM lit tout l'email).</p>
  </form>

  <section class="card hidden" id="result" aria-live="polite">
    <div id="warning" class="banner hidden"></div>
    <span class="verdict" id="verdict"></span>
    <div class="bar" aria-hidden="true"><span id="bar"></span></div>
    <div class="hint" id="scoreText"></div>
    <dl>
      <dt>Raison</dt><dd id="reason"></dd>
      <dt>Avis du ML</dt><dd id="ml"></dd>
      <dt>Mode</dt><dd id="mode"></dd>
      <dt>Durée</dt><dd id="secs"></dd>
    </dl>
  </section>

  <footer>Aide à la décision, pas un verdict définitif : en cas de doute, ne cliquez pas et vérifiez auprès de l'expéditeur par un autre canal.</footer>
</main>

<script>
const $ = (id) => document.getElementById(id);
const EX = {
  ex1: ["Verify your mailbox to avoid suspension", "Your email account will be deactivated within 12 hours because of a pending security upgrade. Confirm your login details here: https://mailbox-keepactive.example-webmail.net/verify Thank you, Webmail Administrator"],
  ex2: ["Team lunch on Thursday", "Hi all, I am booking a table for 12:30 on Thursday near the office. Please tell me by Wednesday evening if you are coming and if you have any dietary restrictions. See you there, Sam"],
};
for (const k of ["ex1", "ex2"]) $(k).onclick = () => { $("subject").value = EX[k][0]; $("body").value = EX[k][1]; };

function show(r) {
  // textContent partout : la "raison" vient d'un LLM qui a lu un email non fiable.
  const v = $("verdict");
  v.textContent = r.is_phishing ? "Phishing probable" : "Email légitime probable";
  v.className = "verdict " + (r.is_phishing ? "phishing" : "legitime");
  $("bar").style.width = Math.round(r.score * 100) + "%";
  $("scoreText").textContent = "Score de phishing : " + Math.round(r.score * 100) + " %";
  $("reason").textContent = r.reason || "—";
  $("ml").textContent = (r.ml_is_phishing ? "phishing" : "légitime") + " (probabilité " + Math.round(r.p_ml * 100) + " %)";
  $("mode").textContent = r.mode === "llm" ? "LLM (vérifié)" : r.mode === "ml_degrade" ? "ML seul (dégradé)" : "ML seul";
  $("secs").textContent = r.seconds + " s";
  const w = $("warning");
  w.textContent = r.avertissement || "";
  w.classList.toggle("hidden", !r.avertissement);
  $("result").classList.remove("hidden");
}

$("form").onsubmit = async (e) => {
  e.preventDefault();
  const btn = $("go"), st = $("status");
  btn.disabled = true; st.className = ""; st.textContent = "Analyse en cours…";
  try {
    const res = await fetch("/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject: $("subject").value, body: $("body").value }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const d = Array.isArray(err.detail) ? err.detail.map((x) => x.msg).join(" ; ") : err.detail;
      throw new Error(d || "Erreur " + res.status);
    }
    show(await res.json());
    st.textContent = "";
  } catch (err) {
    st.className = "bad"; st.textContent = "Échec : " + err.message;
  } finally { btn.disabled = false; }
};
</script>
</body>
</html>
'''