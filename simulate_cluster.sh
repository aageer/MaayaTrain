#!/usr/bin/env bash
# ===========================================================================
#  simulate_cluster.sh — MaayaTrain Local Cluster Simulation
#
#  Simulates a distributed MaayaTrain training cluster on a single machine.
#  Starts a coordinator + 2 workers, all training on the same dataset.
#
#  Usage:
#      chmod +x simulate_cluster.sh
#      ./simulate_cluster.sh
#
#  Then open the dashboard (if enabled) at: http://localhost:8471
#  Press Ctrl+C to stop all processes.
# ===========================================================================

set -euo pipefail

DATA_DIR="./data"
DATA_FILE="${DATA_DIR}/sample_text.txt"
COORD_PORT=7471
DASHBOARD_PORT=8471

# Colours
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  ⚡ MaayaTrain — Local Cluster Simulation               ║${NC}"
echo -e "${CYAN}║  Coordinator + 2 Workers on localhost                   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Ensure sample data exists ──────────────────────────────────
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}Creating sample training data...${NC}"
    mkdir -p "$DATA_DIR"
    python3 -c "
import random, string
random.seed(42)
words = ['the', 'model', 'training', 'distributed', 'compute', 'gradient',
         'network', 'worker', 'parameter', 'optimization', 'learning', 'data',
         'batch', 'loss', 'convergence', 'synchronization', 'quantization',
         'compression', 'tensor', 'weights', 'momentum', 'architecture',
         'transformer', 'attention', 'embedding', 'language', 'generation',
         'inference', 'latency', 'throughput', 'bandwidth', 'protocol']
lines = []
for _ in range(5000):
    line = ' '.join(random.choices(words, k=random.randint(10, 30)))
    lines.append(line)
with open('$DATA_FILE', 'w') as f:
    f.write('\n'.join(lines))
print(f'Created {len(lines)} lines of sample text.')
"
    echo -e "${GREEN}✓ Sample data created: ${DATA_FILE}${NC}"
fi

# ── Step 2: Track PIDs for cleanup ─────────────────────────────────────
PIDS=()
cleanup() {
    echo -e "\n${YELLOW}Shutting down cluster...${NC}"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo -e "${GREEN}✓ All processes stopped.${NC}"
}
trap cleanup EXIT INT TERM

# ── Step 3: Start Coordinator ──────────────────────────────────────────
echo -e "${GREEN}Starting Coordinator (port ${COORD_PORT})...${NC}"
python3 -m maayatrain start \
    --model gpt2-small \
    --dataset "$DATA_FILE" \
    --port "$COORD_PORT" \
    --dashboard \
    --max-steps 2000 &
PIDS+=($!)
sleep 5

# ── Step 4: Start Workers ─────────────────────────────────────────────
echo -e "${GREEN}Starting Worker 1...${NC}"
python3 -m maayatrain join "localhost:${COORD_PORT}" \
    --dataset "$DATA_FILE" &
PIDS+=($!)
sleep 2

echo -e "${GREEN}Starting Worker 2...${NC}"
python3 -m maayatrain join "localhost:${COORD_PORT}" \
    --dataset "$DATA_FILE" &
PIDS+=($!)

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Cluster running: 1 Coordinator + 2 Workers${NC}"
echo -e "${CYAN}  Dashboard:${NC} http://localhost:${DASHBOARD_PORT}"
echo -e "${CYAN}  Telemetry:${NC} ./checkpoints/telemetry.csv"
echo -e "${YELLOW}  Press Ctrl+C to stop.${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""

# Wait for all background processes
wait
