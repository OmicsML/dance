First create a sweep and then optionally spawn multiple agents given the sweep id.

```bash
$ python main.py
```

Record the sweep id ("\[\*\] Sweep ID: \<sweep_id>") and use it to spawn another agent

```bash
$ python main.py --sweep_id <sweep_id>
```

### Known issue

currently there seem to be an issue with wandb sweep agent throwing segfault at the end of the sweep.
This error might carry over to new runs. To fix this, user need to remove the old data and redownload from scratch.
