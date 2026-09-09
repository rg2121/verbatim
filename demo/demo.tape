# A sixty-second demonstration, recorded with VHS (https://github.com/charmbracelet/vhs):
#
#     brew install vhs && vhs demo/demo.tape
#
# Everything here runs against sample/corpus.db, the forty-paper corpus
# committed to this repository, so the recording reproduces on a clean clone.
# `ask` needs a model through Ollama; every other step needs nothing.
Output demo/demo.gif

Set Shell zsh
Set FontSize 15
Set Width 1300
Set Height 800
Set Padding 18
Set TypingSpeed 25ms

Type "# forty papers, their passages, and their vectors - already in the repo"
Enter
Type "uv run verbatim stats --state sample/corpus.db"
Enter
Sleep 4s

Type "clear"
Enter
Type "# every passage carries where it came from"
Enter
Type `uv run verbatim search "how can a model be finetuned on one GPU" --state sample/corpus.db --limit 2`
Enter
Sleep 6s

Type "clear"
Enter
Type "# the point of the project: every quote is checked against its passage"
Enter
Type `uv run verbatim ask "what is QLoRA?" --state sample/corpus.db`
Enter
Sleep 25s

Type "clear"
Enter
Type "# the bibliographies form a graph the corpus can be asked about"
Enter
Type "uv run verbatim graph --state sample/corpus.db --top 3"
Enter
Sleep 5s

Type "clear"
Enter
Type "# and every command records what it did"
Enter
Type "uv run verbatim runs --state sample/corpus.db"
Enter
Sleep 6s
