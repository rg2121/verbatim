# A sixty-second demonstration, recorded with VHS (https://github.com/charmbracelet/vhs):
#
#     brew install vhs && vhs demo/demo.tape
#
# Everything here runs against sample/corpus.db, the forty-paper corpus
# committed to this repository, so the recording reproduces on a clean clone.
# `ask` needs a model through Ollama; every other step needs nothing. `ask`
# writes a ledger row into the sample, so restore it afterwards with
# `git restore sample/corpus.db`.
#
# vhs 0.12 with ffmpeg 9 recorded every frame here and then wrote no GIF. If
# that happens, change the Output line to `Output demo/frames/` and encode:
#
#     ffmpeg -framerate 50 -i demo/frames/frame-text-%05d.png \
#       -framerate 50 -i demo/frames/frame-cursor-%05d.png \
#       -filter_complex "[0][1]overlay,fps=10,pad=w=iw+36:h=ih+36:x=18:y=18:color=0x171717,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
#       demo/demo.gif
Output demo/demo.gif

Set Shell zsh
Set FontSize 15
Set Width 1300
Set Height 800
Set Padding 18
Set TypingSpeed 25ms

# Off camera: let zsh read the commentary lines as comments, and keep the
# stage events that go to stderr out of a recording meant for people.
Hide
Type "setopt interactivecomments; export VERBATIM_LOG_LEVEL=warning; clear"
Enter
Sleep 500ms
Show

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
