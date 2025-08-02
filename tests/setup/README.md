1. Create self-hosted runer. Follow the instructions to config runner.
1. Move `run.sb` under the action runner folder.
1. Create `slurm_history` folder under the action runner folder
1. Finally, submit job via `sbatch run.sb`

### Setting up scheduled jobs via cron

Configure `cron` with the following using `crontab -e`

```bash
# Auto DANCE tests every Sunday, Tuesday, and Thursday night https://github.com/OmicsML/dance
59 23 * * 0,2,4 cd /path/to/runnder && sbatch run.sb
```
