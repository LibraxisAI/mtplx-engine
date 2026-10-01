# A heredoc python program after a pipe reads the piped rows as its source, so a runner silently executed log lines. Give python a file or argv, never both stdin roles

**Symptom:** The depth-policy A/B runner (run_C15.sh, 2026-09-06 04:50) wrote
no result rows and its log ended in `NameError: name 'false' is not defined`
after every arm, although each arm had decoded normally.

**Cause:** The row extractor was written as
`tail -n 1 request-log.jsonl | python3 - "$arm" <<'PY' ... PY`. Both the pipe
and the heredoc are stdin. The shell wires the pipe last, so python read its
program from the pipe: the JSON request-log row, whose `false` literal is not
Python. The heredoc program never ran. Forty minutes of GPU time produced
receipts only because the request log still held every row and the numbers
were recovered from it by hand.

**Fix / rule:** A python one-off that consumes piped data lives in its own
file (`c15_row.py`) and takes the data on stdin or the path as argv. Never
combine `| python3 -` with a heredoc. Before an unattended runner starts a
GPU arm, run its extractor once on a fake row and confirm a result line
appears.
