First create a sweep and then optionally spawn multiple agents given the sweep id.

```bash
$ python main.py
```

Record the sweep id ("\[\*\] Sweep ID: \<sweep_id>") and use it to spawn another agent

```bash
$ python main.py --sweep_id <sweep_id>
```

### Known issue

Currently there seem to be an issue with wandb sweep agent *might be* throwing segfault at the end of the sweep.
This error might carry over to new runs. To fix this, user need to remove the old data and redownload from scratch.

\[Update 2023-06-04\] The segfault seems to happen when writing to the source code
(even with "no changes", e.g., adding blank lines) when runing the sweep agent?..
