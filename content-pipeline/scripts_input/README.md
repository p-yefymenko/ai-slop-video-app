Show scripts now live at `content-pipeline/shows/<id>/script.json`.

This folder remains so an older checkout can still be read. If `<id>.json` is the only copy, the pipeline uses it and prints a warning. When `shows/<id>/script.json` also exists, that file wins and the copy here is ignored.
