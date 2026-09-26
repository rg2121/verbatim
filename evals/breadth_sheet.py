"""Build the blind scoring sheet for the breadth evaluation.

For every question the three answers appear as A, B and C, in an order
shuffled by a secret random seed. Which arm wrote which letter goes to a
mapping file kept outside the repository, and a blind manifest holding its
SHA-256 (and the answers') is written for committing BEFORE any scoring.
evals/breadth_score.py refuses to unblind unless the scores were committed
after that manifest and the mapping still matches its hash.

The first version shuffled with a seed written here, so anyone holding the
repository could rebuild the mapping; the manifest records that amendment.

Declined answers are shown in one neutral wording. Each arm declines in its
own words - the team's sentence is unmistakable - so showing them as written
would unblind the sheet on exactly the answers where it matters most.

The sheet is a single local HTML file with no network dependencies. Scores
save in the browser as you go, and a button downloads them as labels.json.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
import random
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breadth_score import (  # noqa: E402
    ARMS,
    MANIFEST_PATH,
    MAPPING_PATH,
    load_rows,
    question_id,
)

from verbatim import config  # noqa: E402
from verbatim.corpus import CorpusStore  # noqa: E402

NEUTRAL_DECLINE = "No answer - no passage could be cited for this question."
AMENDMENT = (
    "Amendment to evals/breadth/PROTOCOL.md, made 2026-09-26 before any score "
    "existed. The protocol shuffled answer order with a fixed seed written in "
    "the code, so the mapping could be rebuilt from the repository. The order "
    "is now shuffled with a secret random seed, the mapping is kept outside the "
    "repository, and this manifest - committed before scoring - holds the "
    "SHA-256 of the mapping and of the answers. breadth_score.py refuses to "
    "unblind unless the scores were committed after this file and the revealed "
    "mapping and answers match these hashes. The decision rule is unchanged. "
    "Limit: whoever generates the mapping can still open it; blinding against "
    "that rests on procedure unless someone else generates and holds it."
)


def new_seed() -> str:
    """A seed nobody can read from the code, drawn fresh for each blinding."""
    return secrets.token_hex(16)


def blind_manifest(mapping_bytes: bytes, answers_bytes: bytes, *, questions: int) -> dict:
    """What gets committed before scoring: hashes, never the mapping itself."""
    mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
    return {
        "mapping_sha256": mapping_hash,
        "answers_sha256": hashlib.sha256(answers_bytes).hexdigest(),
        "questions": questions,
        # Tied to this blinding, so a re-blinded sheet cannot pick up scores
        # saved in the browser against an earlier mapping.
        "storage_key": f"breadth-labels-{mapping_hash[:12]}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "amendment": AMENDMENT,
    }


def build(rows: dict, questions: list[dict], titles: dict[str, str], seed: str):
    items, mapping = [], {}
    for case in questions:
        question = case["question"]
        arms = rows.get(question, {})
        if set(arms) != set(ARMS):
            continue
        order = list(ARMS)
        random.Random(f"{seed}:{question}").shuffle(order)
        letters = dict(zip("ABC", order, strict=True))
        qid = question_id(question)
        mapping[qid] = letters
        answers = []
        for letter, arm in letters.items():
            answer = arms[arm]["answer"]
            declined = bool(answer["declined"])
            answers.append({
                "letter": letter,
                "text": NEUTRAL_DECLINE if declined else answer["text"],
                "refs": [] if declined else [
                    {"n": n, "title": c["title"], "id": c["arxiv_id"], "quote": c["quoted_text"]}
                    for n, c in enumerate(answer["citations"], start=1)
                ],
            })
        items.append({
            "id": qid,
            "question": question,
            "papers": [titles.get(p, p) for p in case["require_papers"]],
            "answers": answers,
        })
    return items, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("evals/breadth/questions.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("evals/results/breadth_answers.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/results/breadth_sheet.html"))
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    if args.manifest.exists():
        # A new mapping would orphan the committed one and every score made
        # against it. Re-blinding is a decision, so it takes a deletion.
        raise SystemExit(
            f"error: {args.manifest} already exists. Delete it deliberately to "
            "re-blind, and discard any scores made against the old sheet."
        )

    questions = [
        json.loads(line) for line in args.questions.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    rows = load_rows(args.answers)
    with CorpusStore(config.STATE_PATH) as corpus:
        ids = sorted({p for case in questions for p in case["require_papers"]})
        titles = {pid: paper.title for pid, paper in corpus.papers_by_id(ids).items()}
    seed = new_seed()
    items, mapping = build(rows, questions, titles, seed)
    if len(items) < len(questions):
        print(f"note: {len(questions) - len(items)} question(s) lack all three answers and are left off the sheet")
    mapping_bytes = (json.dumps({"seed": seed, "mapping": mapping}, indent=2) + "\n").encode()
    manifest = blind_manifest(mapping_bytes, args.answers.read_bytes(), questions=len(items))

    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.write_bytes(mapping_bytes)
    os.chmod(args.mapping, 0o600)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    args.out.write_text(render_page(items, storage_key=manifest["storage_key"]), encoding="utf-8")

    print(f"{len(items)} question(s) -> {args.out}")
    print(f"Blinding key -> {args.mapping} (outside the repository; do not open it)")
    print(f"Blind manifest -> {args.manifest}")
    print("\n1. Commit the manifest now, BEFORE scoring anything.")
    print("2. Open the sheet in a browser, score every answer, download labels.json.")
    print("3. Save it as evals/breadth/labels.json and commit it.")
    print("4. Run evals/breadth_score.py.")
    return 0


def render_page(items: list[dict], storage_key: str = "breadth-labels") -> str:
    """The sheet, with the answers embedded as data.

    Answer text comes from a model and is untrusted: a closing script tag inside
    one would end the script block and run whatever followed it. Escaping every
    "</" keeps the embedded JSON inert.
    """
    data = json.dumps(items).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data).replace("__KEY__", json.dumps(storage_key))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breadth Scoring Sheet</title>
<style>
  :root{
    --ground:#F4F6F8; --surface:#FFFFFF; --ink:#14181D; --muted:#5D6772;
    --rule:#DAE0E6; --accent:#2E5C8A; --picked:#E3ECF5; --done:#2F7A55;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{ --ground:#0F1317; --surface:#171C21; --ink:#E5E9ED; --muted:#96A1AC;
      --rule:#28313A; --accent:#80B1DF; --picked:#1B2B3B; --done:#6CC79A; }
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--ground); color:var(--ink); font:15px/1.55 var(--sans)}
  header{position:sticky; top:0; z-index:2; background:var(--surface);
    border-bottom:1px solid var(--rule); padding:.7rem 1.2rem;
    display:flex; flex-wrap:wrap; gap:.6rem 1.2rem; align-items:center}
  header h1{font:600 1rem var(--sans); margin:0}
  .progress{font:.85rem var(--mono); color:var(--muted); font-variant-numeric:tabular-nums}
  .spacer{flex:1}
  button{font:inherit; border:1px solid var(--rule); background:var(--surface); color:var(--ink);
    border-radius:6px; padding:.35rem .75rem; cursor:pointer}
  button:hover{border-color:var(--accent)}
  button:focus-visible,input:focus-visible+span{outline:2px solid var(--accent); outline-offset:2px}
  button.primary{background:var(--accent); border-color:var(--accent); color:var(--surface)}
  main{max-width:84rem; margin:0 auto; padding:1.2rem; display:flex; flex-direction:column; gap:1.1rem}
  .rubric{background:var(--surface); border:1px solid var(--rule); border-radius:8px; padding:.2rem 1rem}
  .rubric summary{cursor:pointer; padding:.6rem 0; font-weight:600}
  .rubric table{border-collapse:collapse; width:100%; margin-bottom:.8rem; font-size:.9rem}
  .rubric th,.rubric td{border-top:1px solid var(--rule); padding:.4rem .5rem; text-align:left; vertical-align:top}
  .nav{display:flex; flex-wrap:wrap; gap:.35rem}
  .nav button{min-width:2.4rem; padding:.25rem .4rem; font:.8rem var(--mono)}
  .nav button.current{border-color:var(--accent); color:var(--accent)}
  .nav button.complete{color:var(--done)}
  .question{background:var(--surface); border:1px solid var(--rule); border-radius:8px; padding:1rem 1.2rem}
  .question .k{font:.72rem var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
  .question h2{font:600 1.3rem/1.35 var(--serif); margin:.3rem 0 .5rem; text-wrap:balance}
  .question p{margin:0; color:var(--muted); font-size:.9rem}
  .answers{display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(22rem,1fr))}
  .answer{background:var(--surface); border:1px solid var(--rule); border-radius:8px; padding:1rem;
    display:flex; flex-direction:column; gap:.8rem}
  .answer h3{margin:0; font:600 .95rem var(--mono); color:var(--accent)}
  .text{white-space:pre-wrap; max-width:65ch}
  details.refs summary{cursor:pointer; color:var(--muted); font-size:.85rem}
  .refs ol{margin:.5rem 0 0; padding-left:1.3rem; font-size:.85rem}
  .refs li{margin-bottom:.45rem}
  .refs q{display:block; color:var(--muted)}
  .scores{display:flex; flex-direction:column; gap:.45rem; border-top:1px solid var(--rule); padding-top:.7rem}
  .row{display:flex; align-items:center; justify-content:space-between; gap:.6rem}
  .row .name{font-size:.88rem}
  .chips{display:flex; gap:.3rem}
  .chips label{position:relative}
  .chips input{position:absolute; opacity:0; inset:0; margin:0; cursor:pointer}
  .chips span{display:inline-block; min-width:2.2rem; text-align:center; padding:.2rem .5rem;
    border:1px solid var(--rule); border-radius:5px; font:.85rem var(--mono)}
  .chips input:checked+span{background:var(--picked); border-color:var(--accent); color:var(--accent); font-weight:600}
  .pager{display:flex; justify-content:space-between; gap:1rem}
  @media (prefers-reduced-motion:no-preference){ .answer{transition:border-color .15s} }
</style>
</head>
<body>
<header>
  <h1>Breadth scoring sheet</h1>
  <span class="progress" id="progress"></span>
  <span class="spacer"></span>
  <button class="primary" id="download" type="button">Download labels.json</button>
</header>
<main>
  <details class="rubric" open>
    <summary>How to score - each answer, each row, 0 / 1 / 2. You are not told which system wrote which answer.</summary>
    <table>
      <tr><th></th><th>0</th><th>1</th><th>2</th></tr>
      <tr><th>Coverage</th><td>addresses at most one of the question's two subjects</td><td>both, one only in passing</td><td>both, substantively</td></tr>
      <tr><th>Balance</th><td>one-sided</td><td>a second position mentioned without support</td><td>each position backed by its own citation</td></tr>
      <tr><th>Limitations</th><td>none</td><td>a limitation stated without a citation</td><td>a limitation or counterpoint backed by a citation</td></tr>
    </table>
  </details>
  <nav class="nav" id="nav" aria-label="Questions"></nav>
  <section class="question" id="question"></section>
  <section class="answers" id="answers"></section>
  <div class="pager">
    <button id="prev" type="button">&larr; Previous</button>
    <button id="next" type="button">Next &rarr;</button>
  </div>
</main>
<script>
const ITEMS = __DATA__;
const CRITERIA = [["coverage","Coverage"],["balance","Balance"],["limitations","Limitations"]];
const KEY = __KEY__;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
let current = 0;

function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function complete(item){
  return item.answers.every(a => CRITERIA.every(([c]) => Number.isInteger(labels[item.id]?.[a.letter]?.[c])));
}
function el(tag, cls, text){ const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }

function renderNav(){
  const nav = document.getElementById("nav"); nav.replaceChildren();
  ITEMS.forEach((item, i) => {
    const b = el("button", (i === current ? "current " : "") + (complete(item) ? "complete" : ""), String(i + 1));
    b.type = "button"; b.title = complete(item) ? "scored" : "not finished";
    b.onclick = () => { current = i; render(); };
    nav.appendChild(b);
  });
  const done = ITEMS.filter(complete).length;
  document.getElementById("progress").textContent = `${done} of ${ITEMS.length} questions fully scored`;
}

function render(){
  const item = ITEMS[current];
  const q = document.getElementById("question"); q.replaceChildren(
    el("div", "k", `Question ${current + 1} of ${ITEMS.length}`),
    el("h2", null, item.question),
    el("p", null, "Written from: " + item.papers.join("  ·  ")));
  const box = document.getElementById("answers"); box.replaceChildren();
  item.answers.forEach(a => {
    const card = el("article", "answer");
    card.appendChild(el("h3", null, "Answer " + a.letter));
    card.appendChild(el("div", "text", a.text));
    if (a.refs.length){
      const d = el("details", "refs"); d.appendChild(el("summary", null, `${a.refs.length} reference(s)`));
      const ol = el("ol");
      a.refs.forEach(r => { const li = el("li", null, `[${r.n}] ${r.title} (${r.id})`); const qq = el("q", null, r.quote); li.appendChild(qq); ol.appendChild(li); });
      d.appendChild(ol); card.appendChild(d);
    }
    const scores = el("div", "scores");
    CRITERIA.forEach(([c, name]) => {
      const row = el("div", "row"); row.appendChild(el("span", "name", name));
      const chips = el("div", "chips"); chips.setAttribute("role", "radiogroup"); chips.setAttribute("aria-label", `${name}, answer ${a.letter}`);
      [0,1,2].forEach(v => {
        const label = el("label"); const input = document.createElement("input");
        input.type = "radio"; input.name = `${item.id}-${a.letter}-${c}`; input.value = v;
        input.checked = labels[item.id]?.[a.letter]?.[c] === v;
        input.onchange = () => { ((labels[item.id] ||= {})[a.letter] ||= {})[c] = v; save(); renderNav(); };
        label.append(input, el("span", null, String(v))); chips.appendChild(label);
      });
      row.appendChild(chips); scores.appendChild(row);
    });
    card.appendChild(scores); box.appendChild(card);
  });
  document.getElementById("prev").disabled = current === 0;
  document.getElementById("next").disabled = current === ITEMS.length - 1;
  renderNav();
}

document.getElementById("prev").onclick = () => { if (current > 0){ current--; render(); window.scrollTo(0,0); } };
document.getElementById("next").onclick = () => { if (current < ITEMS.length - 1){ current++; render(); window.scrollTo(0,0); } };
document.getElementById("download").onclick = () => {
  // The scores name their sheet: another blinding orders A, B and C differently.
  const blob = new Blob([JSON.stringify({blinding: KEY, ...labels}, null, 2) + "\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "labels.json";
  document.body.appendChild(a); a.click(); a.remove();
};
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
