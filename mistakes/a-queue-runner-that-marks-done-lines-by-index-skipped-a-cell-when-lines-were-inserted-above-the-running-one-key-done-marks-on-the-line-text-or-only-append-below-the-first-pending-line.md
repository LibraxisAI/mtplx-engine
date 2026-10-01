# A queue runner that marks done lines by index skipped a cell when lines were inserted above the running one — key done marks on the line text, or only append below the first pending line

**Symptom (2026-09-18 04:20):** The overnight GPU queue finished the cell
`p3b-chain-A1`, and the prefill-profile cell that followed it never ran. The
done file held the right count of indices and the log showed no error.

**Cause:** `ov_queue.py` records a finished line by its position in
`queue.txt`. New lines were inserted above a line that had already started,
so when that job ended its index pointed at a different line, and the done
mark landed on a cell that had not run.

**Fix / rule:** While a job is running, only add lines after the first
pending line, and write the file through a temporary file and
`os.replace`. Editing the text of a pending line is safe. A runner meant to
be edited live should key its done marks on a stable id in the line (the
`--name` value), never on the position.
