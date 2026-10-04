# Imitation Learning in Simulation: ACT on ALOHA with LeRobot

A hands-on learning guide: train and evaluate an **ACT (Action Chunking with Transformers)** policy on the **ALOHA bimanual transfer-cube** task with **Hugging Face LeRobot**, on a single laptop GPU. Every step explains what to run, why it exists, how it works underneath, and how to reason about the results.

Verified against LeRobot v0.6.2 and gym-aloha (October 2026). Flags change between LeRobot versions: if one is rejected, run the command with `--help`.

## What you will have at the end

- A working LeRobot + MuJoCo setup on your own GPU machine.
- A baseline success rate from the official pretrained ACT checkpoint.
- Your own trained ACT policy, evaluated on 50 seeded episodes, with rollout videos.
- At least one controlled ablation, and a failure analysis you can explain from first principles.

## Quick start

```bash
# 1. Environment (details: docs/04-setup.md)
conda create -y -n lerobot python=3.12 && conda activate lerobot
conda install -y -c conda-forge ffmpeg libstdcxx-ng
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib && conda deactivate && conda activate lerobot
pip install 'lerobot[training,aloha]'
export MUJOCO_GL=egl

# 2. Check data and simulator
python scripts/smoke_env.py
python scripts/inspect_data.py

# 3. Baseline: migrate the official checkpoint, then evaluate it
python -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir outputs/act_baseline_migrated
lerobot-eval --policy.path=outputs/act_baseline_migrated --policy.device=cuda \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 --eval.batch_size=4 --eval.use_async_envs=false \
  --output_dir=outputs/eval/baseline

# 4. Train your own (≈1 h on an RTX 5080 laptop; keep the charger plugged in)
lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps=30000 --batch_size=8 --num_workers=8 \
  --env_eval_freq=10000 --eval.n_episodes=10 --eval.batch_size=4 --eval.use_async_envs=false \
  --save_freq=10000 --output_dir=outputs/train/act_cube_30k --job_name=act_cube_30k --wandb.enable=false

# 5. Evaluate and analyse
lerobot-eval --policy.path=outputs/train/act_cube_30k/checkpoints/last/pretrained_model --policy.device=cuda \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 --eval.batch_size=4 --eval.use_async_envs=false \
  --output_dir=outputs/eval/act_cube_30k
python scripts/analyse_eval.py outputs/eval/act_cube_30k/eval_info.json
```

## Contents

**Understand**

1. [Concepts: the big picture](docs/01-concepts.md): behaviour cloning, compounding error, multimodality
2. [Environment and data](docs/02-environment-and-data.md): ALOHA task, reward stages, LeRobotDataset format, what the stats show
3. [The model: ACT](docs/03-model-act.md): architecture, CVAE, loss, chunking, temporal ensembling, defaults

**Do**

4. [Setup (Steps 0–2)](docs/04-setup.md): machine check, laptop/GPU notes, install, smoke test
5. [Data inspection and baseline (Steps 3–4)](docs/05-data-and-baseline.md): inspect, watch demos, migrate and evaluate the official model
6. [Training (Steps 5–6)](docs/06-training.md): train command, expected speed, slow-run diagnosis, resuming, proper evaluation

**Go deeper**

7. [Training in depth](docs/07-training-in-depth.md): loss terms, every log field, every setting, checkpoints
8. [Evaluation in depth](docs/08-evaluation-in-depth.md): how evaluation runs, where the reward is defined, every flag and metric
9. [Analysing results (Step 7)](docs/09-analysis.md): failure analysis, statistics, comparing runs
10. [Ablations](docs/10-ablations.md): execution horizon, temporal ensembling, data size, chunking, CVAE
11. [Troubleshooting](docs/11-troubleshooting.md): every error met along the way, with fixes

**Beyond**

12. [Sim to real, and ACT to VLAs](docs/12-sim-to-real-and-vla.md)
13. [Worked example: one complete run](docs/13-worked-example.md): real numbers, findings, write-up, worksheet
14. [Self-check questions](docs/14-self-check.md): 38 questions with answers
15. [Terminology](docs/15-terminology.md): every term in the ACT diagram explained simply and technically, plus a glossary
16. [Config and log reference](docs/16-config-reference.md): every field in the printed train/eval configs and every startup log line

## Scripts

| Script | Purpose | Used in |
| --- | --- | --- |
| [`scripts/smoke_env.py`](scripts/smoke_env.py) | Create the env, print shapes, save a random-policy video | Step 2 |
| [`scripts/inspect_data.py`](scripts/inspect_data.py) | Dataset size, features, stats, chunk and padding check | Step 3 |
| [`scripts/export_episode.py`](scripts/export_episode.py) | Export one demonstration to MP4 | Step 3 |
| [`scripts/analyse_eval.py`](scripts/analyse_eval.py) | Success rate, failure-stage histogram, failed episodes | Step 7 |
| [`scripts/gpu_bench.py`](scripts/gpu_bench.py) | Sanity-check GPU training speed | Step 0 / slow runs |

## Reference results (RTX 5080 laptop, 50 seeded episodes)

| Run | Success | Main failure |
| --- | --- | --- |
| Official checkpoint (80k steps) | 72% | lifted, no hand-over |
| Own ACT, 30k steps | 50% | lifted, then dropped |
| Own ACT, 30k, `n_action_steps=20` | 46% | held without hand-over |

Details and interpretation: [Worked example](docs/13-worked-example.md).

## Three fixes you will almost certainly need

1. **`--eval.use_async_envs=false`** on every `lerobot-eval` and `lerobot-train`, or evaluation crashes with `NamespaceNotFound: gym_aloha`.
2. **Migrate the official checkpoint** before evaluating it (`ProcessorMigrationError`).
3. **Laptops: plug in the charger.** On battery the GPU can sit at P8 / 180 MHz and training runs about 15× slower.

## Repository layout

```
README.md
docs/      16 chapters (linked above)
scripts/   helper scripts
outputs/   created by LeRobot: train/ (checkpoints), eval/ (eval_info.json, videos)
```

## References

- Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, ALOHA). [arXiv 2304.13705](https://arxiv.org/abs/2304.13705)
- ALOHA project page: <https://tonyzhaozh.github.io/aloha>
- LeRobot documentation: <https://huggingface.co/docs/lerobot>
- LeRobot source: <https://github.com/huggingface/lerobot>
- gym-aloha source: <https://github.com/huggingface/gym-aloha>
- Official checkpoint: <https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human>
- Chi et al. *Diffusion Policy*. [arXiv 2303.04137](https://arxiv.org/abs/2303.04137)
- Shukor et al. *SmolVLA*. [arXiv 2506.01844](https://arxiv.org/abs/2506.01844)
