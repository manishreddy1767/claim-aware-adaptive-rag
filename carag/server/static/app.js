// Claim-Aware RAG frontend. All content from the server or from documents is inserted
// as text (never as HTML), so uploaded files cannot inject markup or scripts.

const $ = (id) => document.getElementById(id);

const STATUS_TEXT = {
  SUPPORTED: "Supported",
  PARTIALLY_SUPPORTED: "Partly supported",
  CONTRADICTED: "Contradicted",
  INSUFFICIENT_EVIDENCE: "Not established",
  NOT_SPECIFIED: "Not specified",
  UNCERTAIN: "Uncertain",
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function api(path, { method = "GET", body, form } = {}) {
  const options = { method, headers: {}, credentials: "same-origin" };
  if (form) options.body = form;
  else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("Cannot reach the application. Is it still running?", 0);
  }
  let data = {};
  try { data = await response.json(); } catch { /* empty or non-JSON body */ }
  if (response.status === 401 && !path.startsWith("/api/auth/")) {
    showAuth("Your session has ended. Please sign in again.");
    throw new ApiError(data.error || "Please sign in.", 401);
  }
  if (!response.ok) throw new ApiError(data.error || `Request failed (${response.status}).`, response.status);
  return data;
}

function showError(id, message) {
  const box = $(id);
  box.textContent = message || "";
  box.hidden = !message;
}

let toastTimer;
function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, 3500);
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatWhen(seconds) {
  return new Date(seconds * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/* ---------------------------------------------------------------- auth */

let mode = "local";
let ai = { enabled: false, model_ready: false, model: "" };   // local LLM status from /api/ai
let answerStyle = "quotes";            // "local": saved on this computer; "web": kept only while signed in
let authMode = "login";
let signupAllowed = true;
let authModeChosen = false;
let hasUsers = true;          // whether any account exists yet (first run: none)   // the user picked a tab; a late config reply must not switch it back

function setAuthMode(mode) {
  authMode = mode;
  const register = mode === "register";
  $("tab-login").setAttribute("aria-selected", String(!register));
  $("tab-register").setAttribute("aria-selected", String(register));
  $("auth-title").textContent = register ? "Create your account" : "Sign in";
  $("auth-submit").textContent = register ? "Create account" : "Sign in";
  $("password").setAttribute("autocomplete", register ? "new-password" : "current-password");
  $("password-hint").hidden = !register;
  // The explanation follows the chosen tab, so heading and text always agree.
  $("auth-subtitle").textContent = !hasUsers
    ? (register ? "Create the first account to get started. It is stored only on this computer."
                : "No account exists yet. Choose Create account to make the first one.")
    : (register ? "Create a new account. Each account sees only its own documents and questions."
                : "Answers from your own documents, with every claim checked against the source.");
  showError("auth-error", "");
}

async function showAuth(message) {
  $("app-view").hidden = true;
  $("auth-view").hidden = false;
  stopHealthPolling();
  authModeChosen = false;
  try {
    const config = await api("/api/auth/config");
    signupAllowed = config.allow_signup;
    mode = config.mode;
    $("auth-fineprint").textContent = mode === "web"
      ? "Documents you add are kept only while you are signed in and are deleted when you sign out."
      : "Your account, documents and questions stay on this computer.";
    $("tab-register").hidden = !signupAllowed;
    hasUsers = config.has_users;
    if (!authModeChosen || !signupAllowed) setAuthMode(config.has_users || !signupAllowed ? "login" : "register");
  } catch (error) {
    showError("auth-error", error.message);
  }
  if (message) showError("auth-error", message);
  $("username").focus();
}

async function submitAuth(event) {
  event.preventDefault();
  const username = $("username").value.trim();
  const password = $("password").value;
  if (!username || !password) {
    showError("auth-error", "Enter a username and a password.");
    return;
  }
  const button = $("auth-submit");
  button.disabled = true;
  try {
    const path = authMode === "register" ? "/api/auth/register" : "/api/auth/login";
    const data = await api(path, { method: "POST", body: { username, password } });
    $("password").value = "";
    await showApp(data.username);
  } catch (error) {
    showError("auth-error", error.message);
  } finally {
    button.disabled = false;
  }
}

async function logout() {
  if (mode === "web" && documentCount > 0 &&
      !confirm("Signing out deletes the documents you added and your questions. Sign out?")) return;
  try { await api("/api/auth/logout", { method: "POST" }); } catch { /* signing out anyway */ }
  $("answers").replaceChildren();
  $("check-result").replaceChildren();
  $("welcome").hidden = false;
  showAuth();
}

/* ---------------------------------------------------------------- health */

let healthTimer;
function stopHealthPolling() { clearTimeout(healthTimer); }

async function pollHealth() {
  const badge = $("model-status");
  try {
    const health = await api("/api/health");
    if (health.model_error) {
      badge.textContent = "Models failed to load";
      badge.className = "status-dot failed";
      badge.title = health.model_error;
      return;
    }
    if (health.models_ready) {
      badge.textContent = health.device === "cuda" ? "Ready · GPU" : "Ready · CPU";
      badge.className = "status-dot ready";
      return;
    }
    badge.textContent = "Loading models…";
    badge.className = "status-dot";
  } catch { /* retry below */ }
  healthTimer = setTimeout(pollHealth, 2000);
}

/* ---------------------------------------------------------------- documents */

let documentCount = 0;

function docIcon(name, origin) {
  if (origin === "url") return el("span", { class: "doc-icon", "aria-hidden": "true", text: "WEB" });
  const ext = name.includes(".") ? name.split(".").pop().toUpperCase() : "DOC";
  return el("span", { class: "doc-icon", "aria-hidden": "true", text: ext.slice(0, 4) });
}

function docOrigin(doc) {
  if (doc.origin === "url") {
    let host = doc.location;
    try { host = new URL(doc.location).hostname; } catch { /* keep the raw address */ }
    return `Webpage · ${host} · ${doc.sentences} sentences`;
  }
  if (doc.origin === "path") return `On this computer · ${doc.sentences} sentences`;
  return `${doc.sentences} sentences · ${formatSize(doc.size)}`;
}

async function loadDocuments() {
  const data = await api("/api/documents");
  const list = $("doc-list");
  list.replaceChildren(...data.documents.map((doc) => el("li", {},
    docIcon(doc.name, doc.origin),
    el("div", { class: "doc-meta" },
      el("div", { class: "doc-name", text: doc.name, title: doc.name }),
      el("div", { class: "doc-sub", text: docOrigin(doc), title: doc.location || undefined }),
      ...doc.warnings.map((w) => el("div", { class: "doc-warn", text: w }))),
    el("button", {
      class: "icon-btn", title: `Delete ${doc.name}`, "aria-label": `Delete ${doc.name}`, text: "✕",
      onclick: () => deleteDocument(doc),
    }))));
  documentCount = data.documents.length;
  $("doc-count").textContent = String(documentCount);
  $("doc-empty").hidden = documentCount > 0;
}

async function deleteDocument(doc) {
  if (!confirm(`Delete "${doc.name}"? Questions will no longer use it.`)) return;
  try {
    await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
    toast(`Deleted ${doc.name}`);
    await loadDocuments();
  } catch (error) {
    toast(error.message);
  }
}

async function uploadFiles(files) {
  const uploads = $("uploads");
  for (const file of files) {
    const row = el("li", { text: `Adding ${file.name}…` });
    uploads.append(row);
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const data = await api("/api/documents", { method: "POST", form });
      row.remove();
      toast(`Added ${data.document.name} (${data.document.sentences} sentences)`);
      await loadDocuments();
    } catch (error) {
      row.className = "error";
      row.textContent = error.message;
      setTimeout(() => row.remove(), 8000);
    }
  }
}

async function addFromForm(event, { input, button, path, body, describe }) {
  event.preventDefault();
  const value = $(input).value.trim();
  if (!value) { $(input).focus(); return; }
  const row = el("li", { text: `Adding ${value}…` });
  $("uploads").append(row);
  $(button).disabled = true;
  try {
    const data = await api(path, { method: "POST", body: body(value) });
    row.remove();
    $(input).value = "";
    toast(describe(data));
    await loadDocuments();
  } catch (error) {
    row.className = "error";
    row.textContent = error.message;
    setTimeout(() => row.remove(), 10000);
  } finally {
    $(button).disabled = false;
  }
}

function setUpSourceForms() {
  $("url-form").addEventListener("submit", (event) => addFromForm(event, {
    input: "url-input", button: "url-button", path: "/api/documents/url", body: (url) => ({ url }),
    describe: (data) => `Added ${data.document.name} (${data.document.sentences} sentences)`,
  }));
  $("path-form").addEventListener("submit", (event) => addFromForm(event, {
    input: "path-input", button: "path-button", path: "/api/documents/path", body: (path) => ({ path }),
    describe: (data) => {
      const added = `Added ${data.added.length} document${data.added.length === 1 ? "" : "s"}`;
      if (!data.errors.length) return added;
      const first = data.errors[0];
      return `${added}; ${data.errors.length} skipped (${first.name}: ${first.error})`;
    },
  }));
}

function setUpDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  input.addEventListener("change", () => {
    const files = [...input.files];
    input.value = "";
    if (files.length) uploadFiles(files);
  });
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
  });
  zone.addEventListener("dragover", (event) => { event.preventDefault(); zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
    const files = [...event.dataTransfer.files];
    if (files.length) uploadFiles(files);
  });
}

/* ---------------------------------------------------------------- answers */

// Answer text with "[n]" markers turned into buttons that highlight source n.
function answerText(text, card) {
  const node = el("p", { class: "answer-text" });
  for (const part of text.split(/(\[\d+\])/)) {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) { node.append(part); continue; }
    const marker = match[1];
    node.append(el("button", {
      class: "cite", text: marker, title: `Show source ${marker}`, "aria-label": `Source ${marker}`,
      onclick: () => {
        const source = card.querySelector(`[data-marker="${marker}"]`);
        if (!source) return;
        source.scrollIntoView({ behavior: "smooth", block: "nearest" });
        source.classList.add("flash");
        setTimeout(() => source.classList.remove("flash"), 1200);
      },
    }));
  }
  return node;
}

function sourceWhere(citation) {
  const parts = [citation.source];
  if (citation.section) parts.push(citation.section);
  if (citation.page) parts.push(`page ${citation.page}`);
  return parts.join(" · ");
}

function claimRow(claim) {
  const why = claim.self_supported
    ? "Quoted directly from the document."
    : claim.explanation;
  return el("div", { class: "claim" },
    el("span", { class: `claim-status s-${claim.status}`, text: STATUS_TEXT[claim.status] || claim.status }),
    el("div", {}, el("div", { text: claim.claim }), why ? el("div", { class: "claim-why", text: why }) : null));
}

function answerCard(view) {
  const card = el("article", { class: "card" });
  card.append(
    el("span", { class: `pill o-${view.outcome}`, text: view.outcome_label }),
    view.ai ? el("span", { class: "ai-tag", text: `AI · ${view.ai.model}` }) : null,
    el("p", { class: "card-question", text: view.question }),
    answerText(view.answer, card));

  if (view.citations.length) {
    card.append(el("div", { class: "sources" },
      el("h3", { text: "Sources" }),
      ...view.citations.map((c) => el("div", { class: "source", "data-marker": String(c.marker) },
        el("span", { class: "source-num", text: String(c.marker) }),
        el("span", { class: "source-where", text: sourceWhere(c) }),
        el("span", { class: "source-text", text: c.text })))));
  }

  const checks = [...view.claims.map(claimRow)];
  if (view.ai && view.ai.draft) {
    checks.push(el("p", { class: "claim-why", text: "The model's original answer, before checking:" }),
      el("p", { class: "draft", text: view.ai.draft }));
  }
  if (view.ai && view.removed_claims.length) {
    checks.push(el("p", { class: "claim-why", text: "Removed because the documents do not support them:" }),
      ...view.removed_claims.map((c) => el("p", { class: "claim-why", text: `• ${c.claim}` })));
  } else if (view.removed_claims.length) {
    checks.push(el("p", { class: "claim-why", text:
      `${view.removed_claims.length} candidate sentence(s) were left out because the documents do not support them.` }));
  }
  const notes = view.notes.filter((n) => !n.startsWith("Comparison side"));
  const meta = [];
  if (view.timings) {
    const total = (view.timings.retrieval_s || 0) + (view.timings.verification_s || 0);
    meta.push(`${total.toFixed(1)} s`);
  }
  if (view.budget) meta.push(`${view.budget.used} evidence checks`);
  if (checks.length || notes.length) {
    card.append(el("details", { class: "details" },
      el("summary", { text: "How this answer was checked" }),
      ...checks,
      ...notes.map((n) => el("p", { class: "claim-why", text: n })),
      meta.length ? el("p", { class: "meta", text: meta.join(" · ") }) : null));
  }
  return card;
}

async function ask(event) {
  event.preventDefault();
  const input = $("question");
  const question = input.value.trim();
  showError("ask-error", "");
  if (!question) { showError("ask-error", "Type a question first."); return; }
  if (documentCount === 0) { showError("ask-error", "Add at least one document first."); return; }
  const button = $("ask-button");
  button.disabled = true;
  $("welcome").hidden = true;
  const style = ai.enabled && ai.model_ready ? answerStyle : "quotes";
  const pending = el("div", { class: "card thinking" }, el("span", { class: "spinner" }),
    el("span", { text: style === "ai"
      ? `${ai.model} is writing an answer from your documents; each sentence is then checked. ` +
        "The first answer can take a minute while the model loads."
      : "Searching your documents and checking each claim…" }));
  $("answers").prepend(pending);
  try {
    const view = await api("/api/ask", { method: "POST", body: { question, style } });
    pending.replaceWith(answerCard(view));
    input.value = "";
  } catch (error) {
    pending.remove();
    showError("ask-error", error.message);
    if (!$("answers").children.length) $("welcome").hidden = false;
  } finally {
    button.disabled = false;
    input.focus();
  }
}

/* ---------------------------------------------------------------- check text */

async function checkText(event) {
  event.preventDefault();
  const text = $("check-text").value.trim();
  showError("check-error", "");
  if (!text) { showError("check-error", "Paste some text to check."); return; }
  if (documentCount === 0) { showError("check-error", "Add at least one document first."); return; }
  const button = $("check-button");
  button.disabled = true;
  const result = $("check-result");
  result.replaceChildren(el("div", { class: "card thinking" }, el("span", { class: "spinner" }),
    el("span", { text: "Checking each statement against your documents…" })));
  try {
    const data = await api("/api/verify", { method: "POST", body: { text } });
    result.replaceChildren(el("article", { class: "card" },
      el("span", { class: `pill ${data.abstained ? "o-abstain" : "o-answer"}`,
        text: data.abstained ? "Nothing in this text is established" : "Checked version" }),
      el("p", { class: "answer-text", text: data.revised }),
      el("div", { class: "sources" },
        el("h3", { text: "Statements" }),
        ...data.claims.map((c) => el("div", { class: "claim" },
          el("span", { class: `claim-status s-${c.status}`, text: STATUS_TEXT[c.status] || c.status }),
          el("div", {}, el("div", { text: c.claim }),
            c.evidence.length ? el("div", { class: "claim-why", text: `Source: ${c.evidence[0]}` }) : null)))),
    ));
  } catch (error) {
    result.replaceChildren();
    showError("check-error", error.message);
  } finally {
    button.disabled = false;
  }
}

/* ---------------------------------------------------------------- history */

async function loadHistory() {
  const data = await api("/api/history");
  const list = $("history-list");
  list.replaceChildren(...data.history.map((item) => el("button", {
    class: "history-item",
    onclick: () => {
      selectTab("ask-panel");
      $("welcome").hidden = true;
      $("answers").prepend(answerCard(item.result));
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
  }, el("span", {}, el("span", { class: `pill o-${item.outcome}`, text: item.result.outcome_label }),
       el("div", { class: "history-q", text: item.question })),
     el("span", { class: "history-when", text: formatWhen(item.created_at) }))));
  $("history-empty").hidden = data.history.length > 0;
  $("clear-history").hidden = data.history.length === 0;
}

async function clearHistory() {
  if (!confirm("Clear all earlier questions?")) return;
  try {
    await api("/api/history", { method: "DELETE" });
    await loadHistory();
  } catch (error) {
    toast(error.message);
  }
}

/* ---------------------------------------------------------------- local AI */

function renderStyleSwitch() {
  const available = ai.enabled && ai.model_ready;
  $("style-switch").hidden = !available;
  $("ask-hint").hidden = available;
  $("style-ai").textContent = available ? `AI answer · ${ai.model}` : "AI answer";
  if (!available) answerStyle = "quotes";
  for (const button of $("style-switch").querySelectorAll("button")) {
    button.setAttribute("aria-checked", String(button.dataset.style === answerStyle));
  }
}

async function loadAi() {
  try {
    ai = await api("/api/ai");
  } catch {
    ai = { enabled: false, model_ready: false, model: "" };
  }
  try { if (localStorage.getItem("answerStyle") === "ai") answerStyle = "ai"; } catch { /* storage blocked */ }
  renderStyleSwitch();
}

function renderAiDialog() {
  const status = $("ai-status");
  if (!ai.running) {
    status.textContent = "Ollama is not running on this computer.";
    status.className = "ai-status bad";
    $("ai-help").textContent = "Install Ollama from https://ollama.com/download, start it, then run in a terminal: " +
      `ollama pull ${ai.model || "qwen3:8b"}`;
  } else if (!ai.model_ready) {
    status.textContent = `Ollama is running, but ${ai.model} is not installed.`;
    status.className = "ai-status bad";
    $("ai-help").textContent = `In a terminal, run: ollama pull ${ai.model}   (or choose an installed model).`;
  } else {
    status.textContent = `Ready: ${ai.model}`;
    status.className = "ai-status ok";
    $("ai-help").textContent = "Larger models write better answers but need more memory. On a GPU with less than " +
      "6 GB, an 8B model runs partly on the processor and is slower.";
  }
  const select = $("ai-model");
  const names = [...new Set([...(ai.installed || []), ai.model].filter(Boolean))];
  select.replaceChildren(...names.map((name) => el("option", {
    value: name, text: (ai.installed || []).includes(name) ? name : `${name} (not installed)`,
  })));
  select.value = ai.model;
  $("ai-enabled").checked = ai.enabled;
  for (const id of ["ai-enabled", "ai-model", "ai-save"]) $(id).disabled = !ai.editable;
  if (!ai.editable) $("ai-help").textContent = "AI settings are managed by the operator of this website.";
}

async function openAiDialog() {
  showError("ai-error", "");
  await loadAi();
  renderAiDialog();
  $("ai-dialog").showModal();
}

async function saveAi(event) {
  event.preventDefault();
  showError("ai-error", "");
  try {
    ai = await api("/api/ai", { method: "PUT", body: { enabled: $("ai-enabled").checked, model: $("ai-model").value } });
    renderAiDialog();
    renderStyleSwitch();
    if (ai.enabled && !ai.model_ready) {
      showError("ai-error", "Saved, but the model is not ready yet; AI answers appear once it is installed.");
      return;
    }
    $("ai-dialog").close();
    toast(ai.enabled ? `AI answers on: ${ai.model}` : "AI answers turned off");
  } catch (error) {
    showError("ai-error", error.message);
  }
}

/* ---------------------------------------------------------------- shell */

function selectTab(panelId) {
  for (const tab of $("main-tabs").querySelectorAll("button")) {
    const selected = tab.dataset.panel === panelId;
    tab.setAttribute("aria-selected", String(selected));
    $(tab.dataset.panel).hidden = !selected;
  }
  if (panelId === "history-panel") loadHistory().catch((error) => toast(error.message));
}

async function showApp(username) {
  $("auth-view").hidden = true;
  $("app-view").hidden = false;
  $("user-name").textContent = username;
  try {
    const me = await api("/api/auth/me");
    mode = me.mode;
    $("path-form").hidden = !me.local_files;
  } catch { /* keep defaults */ }
  const note = $("storage-note");
  note.textContent = mode === "web"
    ? "Kept only while you are signed in. Signing out deletes your documents and questions."
    : "Saved on this computer. Your documents stay after you sign out.";
  note.className = mode === "web" ? "storage-note web" : "storage-note";
  selectTab("ask-panel");
  pollHealth();
  loadAi();
  try {
    await loadDocuments();
  } catch (error) {
    toast(error.message);
  }
  $("question").focus();
}

function init() {
  $("auth-tabs").addEventListener("click", (event) => {
    const mode = event.target.dataset?.mode;
    if (mode) { authModeChosen = true; setAuthMode(mode); }
  });
  $("auth-form").addEventListener("submit", submitAuth);
  $("logout").addEventListener("click", logout);
  $("main-tabs").addEventListener("click", (event) => {
    const panel = event.target.dataset?.panel;
    if (panel) selectTab(panel);
  });
  $("ask-form").addEventListener("submit", ask);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("ask-form").requestSubmit();
  });
  $("check-form").addEventListener("submit", checkText);
  $("clear-history").addEventListener("click", clearHistory);
  setUpDropzone();
  setUpSourceForms();
  $("ai-open").addEventListener("click", openAiDialog);
  $("ai-close").addEventListener("click", () => $("ai-dialog").close());
  $("ai-form").addEventListener("submit", saveAi);
  $("style-switch").addEventListener("click", (event) => {
    const style = event.target.dataset?.style;
    if (!style) return;
    answerStyle = style;
    try { localStorage.setItem("answerStyle", style); } catch { /* storage blocked */ }
    renderStyleSwitch();
  });

  api("/api/auth/me")
    .then((me) => showApp(me.username))
    .catch(() => showAuth());
}

init();
