#!/usr/bin/env bash
set -e

# 顶层目录
mkdir -p src/engisynth
mkdir -p configs tests notebooks docs data outputs

# 子包目录
mkdir -p src/engisynth/config
mkdir -p src/engisynth/data
mkdir -p src/engisynth/constraints
mkdir -p src/engisynth/generators
mkdir -p src/engisynth/postprocess
mkdir -p src/engisynth/evaluation
mkdir -p src/engisynth/cli
mkdir -p src/engisynth/utils

# __init__.py
touch src/engisynth/__init__.py
touch src/engisynth/config/__init__.py
touch src/engisynth/data/__init__.py
touch src/engisynth/constraints/__init__.py
touch src/engisynth/generators/__init__.py
touch src/engisynth/postprocess/__init__.py
touch src/engisynth/evaluation/__init__.py
touch src/engisynth/cli/__init__.py
touch src/engisynth/utils/__init__.py

# CLI 脚本
touch src/engisynth/cli/create_config.py
touch src/engisynth/cli/run_pipeline.py
touch src/engisynth/cli/run_tests.py

# 其他顶层文件
touch requirements.txt
touch README.md
cat << 'EOF' > .gitignore
__pycache__/
*.pyc
data/
outputs/
.DS_Store
EOF

# 首次提交
git add .
git commit -m "chore: scaffold EngiSynth project structure"

