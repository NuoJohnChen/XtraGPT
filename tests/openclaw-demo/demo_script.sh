#!/usr/bin/env bash
# Paced shell script used as the payload for asciinema rec.
set -u
export PATH="/home/nuochen/.nvm/versions/node/v24.13.1/bin:$HOME/.local/bin:$PATH"

GREEN='\033[1;32m'
BLUE='\033[1;34m'
YEL='\033[1;33m'
RST='\033[0m'

typed() {
  local prompt="${BLUE}demo@xtragpt${RST} ${GREEN}\$${RST} "
  printf "%b%s" "$prompt" "$1"
  # Simulate typing: newline, then show the command's output
  sleep 0.6
  printf "\n"
}

banner() {
  printf "\n${YEL}── %s ──${RST}\n" "$1"
  sleep 1
}

banner "1. What's in the scaffolded skill pack"
typed "tree -L 3 openclaw"
tree -L 3 openclaw 2>/dev/null || find openclaw -maxdepth 3 -print | sed 's|[^/]*/|  |g;s|  \([^ ]\)|├─ \1|'
sleep 3

banner "2. The skill YAML is the source of truth for the prompt"
typed "head -25 openclaw/skills/skill.xtragpt-paper-revision-skill.yaml"
head -25 openclaw/skills/skill.xtragpt-paper-revision-skill.yaml
sleep 3

banner "3. XtraGPT-7B is live on a local A100 (vLLM)"
typed "curl -s http://127.0.0.1:8088/v1/models | python -m json.tool | head -8"
curl -s http://127.0.0.1:8088/v1/models | python -m json.tool | head -8
sleep 2

banner "4. OpenClaw sees the xtragpt-local provider we added"
typed "openclaw config get models.providers.xtragpt-local"
openclaw config get models.providers.xtragpt-local 2>/dev/null | head -14
sleep 3

banner "5. Run the three-way comparison (XtraGPT-7B vs Qwen2.5-7B base vs GLM-5.1)"
typed "python run_demo.py"
source /home/nuochen/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory >/dev/null 2>&1
python run_demo.py 2>&1 | tail -12
sleep 2

banner "6. The side-by-side report"
typed "sed -n '1,30p' outputs/before_after.md"
sed -n '1,30p' outputs/before_after.md
sleep 2

typed "sed -n '31,82p' outputs/before_after.md | head -40"
sed -n '31,82p' outputs/before_after.md | head -40
sleep 3

banner "That's the skill in action"
printf "XtraGPT-7B: concise, no meta-wrapping, fastest  (%s)\n"    "fine-tuned on 140k paper-revision pairs"
printf "Qwen2.5-7B: same capacity, no revision fine-tune — adds boilerplate '%s'\n" "Revised text:"
printf "GLM-5.1:    general-purpose big model — wordier, 5-12x slower\n"
sleep 3
