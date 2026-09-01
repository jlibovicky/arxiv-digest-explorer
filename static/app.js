"use strict";

const PAGE_SIZE = 30;
const PREFETCH_AT = 5;

const el = (id) => document.getElementById(id);

const queue = [];      // papers not shown yet, best score first
let current = null;    // the paper on the screen
const deferredIds = new Set(); // "remind later" papers, back on the list next run
let pending = 0;       // papers still on the reading list per the server
let exhausted = false; // the server has no more pages for us
let loading = null;

function toast(message) {
  const node = el("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 1600);
}

async function loadPage() {
  if (exhausted || loading) return loading;
  const params = new URLSearchParams({ limit: PAGE_SIZE });
  const last = queue.length ? queue[queue.length - 1] : current;
  if (last) {
    params.set("cursor_score", last.score);
    params.set("cursor_date", last.date);
  }
  loading = fetch(`/api/queue?${params}`)
    .then((response) => response.json())
    .then((data) => {
      pending = data.pending;
      if (!data.papers.length) exhausted = true;
      queue.push(...data.papers.filter((p) => !deferredIds.has(p.arxiv_id)));
    })
    .catch(() => toast("Could not reach the server."))
    .finally(() => { loading = null; });
  return loading;
}

function render() {
  const card = el("card");
  if (!current) {
    card.hidden = true;
    el("empty").hidden = false;
    el("progress").textContent = deferredIds.size
      ? `done — ${deferredIds.size} kept for next time`
      : "done";
    return;
  }

  el("empty").hidden = true;
  card.hidden = false;

  const idLink = el("arxiv-id");
  idLink.textContent = current.arxiv_id;
  idLink.href = current.url;
  el("date").textContent = new Date(current.date)
    .toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  el("score").textContent = `score ${(100 * current.score).toFixed(0)}%`;
  el("title").innerHTML = current.title_html;
  el("authors").textContent = current.authors;
  const comment = el("comment");
  comment.textContent = current.comment || "";
  comment.hidden = !current.comment;
  el("abstract").innerHTML = current.abstract_html;

  const left = Math.max(pending - deferredIds.size, queue.length + 1);
  el("progress").textContent = `${left} to go`;
  window.scrollTo(0, 0);
}

function advance() {
  current = queue.shift() || null;
  if (queue.length <= PREFETCH_AT) loadPage();
  render();
}

async function decide(decision) {
  if (!current) return;
  const paper = current;
  if (decision === "yes") {
    // Opened straight from the key press so the browser allows the new tab.
    window.open(paper.url, "_blank", "noopener");
  }
  pending -= 1;
  advance();
  const response = await fetch("/api/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arxiv_id: paper.arxiv_id, decision }),
  }).catch(() => null);
  if (!response || !response.ok) {
    toast(`Could not log the decision for ${paper.arxiv_id}.`);
  }
}

function later() {
  if (!current) return;
  deferredIds.add(current.arxiv_id);
  advance();
}

async function refresh() {
  const button = el("refresh");
  button.disabled = true;
  button.textContent = "Refreshing…";
  await fetch("/api/refresh", { method: "POST" });
  const poll = setInterval(async () => {
    const status = await fetch("/api/status").then((r) => r.json());
    if (status.refreshing) return;
    clearInterval(poll);
    button.disabled = false;
    button.textContent = "Refresh from arXiv";
    const result = status.last_refresh || {};
    if (result.error) {
      toast(`Refresh failed: ${result.error}`);
    } else {
      toast(`${result.added || 0} new papers on the list.`);
    }
    // The new papers may outrank what we have, so start the list over.
    queue.length = 0;
    exhausted = false;
    current = null;
    await loadPage();
    advance();
  }, 1500);
}

function zoteroOpen() {
  return !el("zotero-overlay").hidden;
}

function escapeHtml(text) {
  const node = document.createElement("div");
  node.textContent = text || "";
  return node.innerHTML;
}

function renderZotero(papers) {
  const body = el("zotero-body");
  if (!papers.length) {
    body.innerHTML = '<p class="empty">The Zotero collection is empty.</p>';
    return;
  }
  body.innerHTML = papers.map((paper) => {
    const title = paper.url
      ? `<a href="${escapeHtml(paper.url)}" target="_blank" rel="noopener">${escapeHtml(paper.title)}</a>`
      : escapeHtml(paper.title);
    const tags = paper.tags
      .map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
    const abstract = paper.abstract
      ? `<p class="abstract">${escapeHtml(paper.abstract)}</p>` : "";
    // The note is HTML the user wrote in Zotero, so it goes in as it is.
    const note = paper.note_html
      ? `<div class="note">${paper.note_html}</div>` : "";
    return `<article class="zotero-item">
      <h3>${title}</h3>
      <p class="authors">${escapeHtml(paper.authors)}</p>
      <p class="tags">${tags}</p>
      ${abstract}
      ${note}
    </article>`;
  }).join("");
}

async function showZotero() {
  const overlay = el("zotero-overlay");
  const again = el("zotero-again");
  overlay.hidden = false;
  again.disabled = true;
  el("zotero-body").innerHTML = '<p class="empty">Asking Zotero…</p>';
  try {
    const response = await fetch("/api/zotero");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    renderZotero(data.papers);
  } catch (error) {
    el("zotero-body").innerHTML =
      `<p class="empty">Could not load Zotero notes: ${escapeHtml(error.message)}</p>`;
  } finally {
    again.disabled = false;
  }
}

function hideZotero() {
  el("zotero-overlay").hidden = true;
}

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (zoteroOpen()) {
    // The reading-list shortcuts stay off while the notes are on the screen.
    if (event.key === "Escape") hideZotero();
    return;
  }
  switch (event.key.toLowerCase()) {
    case "y": decide("yes"); break;
    case "n": decide("no"); break;
    case "l": case " ": event.preventDefault(); later(); break;
    case "z": showZotero(); break;
    default: return;
  }
});

el("yes").addEventListener("click", () => decide("yes"));
el("no").addEventListener("click", () => decide("no"));
el("later").addEventListener("click", later);
el("refresh").addEventListener("click", refresh);
el("zotero").addEventListener("click", showZotero);
el("zotero-again").addEventListener("click", showZotero);
el("zotero-close").addEventListener("click", hideZotero);
el("zotero-overlay").addEventListener("click", (event) => {
  if (event.target === el("zotero-overlay")) hideZotero();
});

loadPage().then(advance);
