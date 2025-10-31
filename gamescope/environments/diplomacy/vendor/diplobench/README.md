# DiplomacyBench

This is an **in-progress** repository for assessing large language models' ability to play **full press Diplomacy** (i.e., with open negotiations and alliances) against other AI agents. The goal is to explore how LLMs reason, negotiate, ally, and betray in a highly strategic competitive game.

So far, this is not fully implemented as a benchmark, but as a framework for LLMs to play a standard full press game of Diplomacy. As of now, 8 LLMs can play against each other, with reporting on the performance of the test model.

It's not really economical to use as a benchmark since each round takes a lot of API calls, and quite a few iterations would be needed to stabilise the scores. So for now this will simply be a testbed for experimenting with LLMs playing diplomacy.

> **Why is this interesting?**  
> Diplomacy is a classic game of negotiation and balance of power that poses unique challenges in both strategy and communication. It's an interesting test-bed for game theory and complex decision making.

We test if modern LLMs can reason about alliances, plan multi-stage maneuvers, and negotiate effectively—**zero-shot**. Surprisingly, some models exhibit coherent strategic thinking and can form (and break) alliances in ways that mirror human play.

Here's a game with sonnet-3.7 playing as Austria (the red player), reaching the win condition after 14 turns:

https://github.com/user-attachments/assets/6a307cec-a889-4d57-a2e0-fbc9e94f63b9

## Overview

### Game Structure

In our test setup, games are run for up to 50 game-years, with a 4-round negotiation phase before each movement turn. The test model we are examining always plays as Austria (AUT), which has a challenging centre start. The competing models are all powered by gemini-2.0-flash-001 and given a character card for personality & play style.

Each **Power** (e.g., France, England, Germany) is controlled by a separate LLM. These LLMs communicate and negotiate with each other through short messages (the “press” or “missives”) multiple times each turn, then decide on final orders (moves). The game engine processes these orders, updates the board state, and continues until the game ends.


## Sample Game Reports & Negotiation Logs

**Note** that these are just single games, so may not be representative of averaged performance over many trials. There will be significant variance of outcomes between iterations. From my testing, these samples are *roughly* indicative of how these models play.

The negotiation logs in these reports provide interesting insight into their different styles of negotiating.

- http://eqbench.com/results/diplomacy/claude-3.7-sonnet-1.54.html
- http://eqbench.com/results/diplomacy/deepseek-r1-1.56.html
- http://eqbench.com/results/diplomacy/gemini-2.0-flash-001-1.51.html
- http://eqbench.com/results/diplomacy/gpt-4o-mini-1.43.html
- http://eqbench.com/results/diplomacy/o1-1.31.html
- http://eqbench.com/results/diplomacy/o3-mini-high-1.45.html

![image](https://github.com/user-attachments/assets/9da1fb4d-2486-479f-947f-5adba5754402)

---

## Repository Contents

- **`main.py`**  
  The primary entry point. It sets up a standard Diplomacy game, creates `LLMAgent` instances (one per power), and runs multiple turns until a conclusion or turn limit.

- **`diplomacy_game/`**  
  Contains core modules for orchestrating the game:
  - `environment.py`: Wrapper around the python-diplomacy `Game` class, with custom BFS-based connectivity logic and state extraction.  
  - `agent.py`: Defines `LLMAgent`, the LLM-based player that negotiates, composes messages, and decides orders.  
  - `llm.py`: Code to call the LLM API.  
  - `recommendation_engine.py`: Wraps the RL-based “welfare” policy and produces recommended moves.  
  - `persistence.py`: Handles saving/loading game states to/from JSON.

- **`welfare_diplomacy_baselines/`**  
  The RL policy code adapted from [Welfare Diplomacy Baselines](https://github.com/hannaherlebach/welfare_diplomacy_baselines). We store policy weights here.

- **`generate-reports.ipynb`**  
  A notebook to parse JSON game logs and produce tabular summaries or final outcomes.

- **`generate-webp.ipynb`**  
  A notebook to generate **animated webp** files showing each phase of the game as it progresses.

- **`requirements.txt`**  
  Lists minimal packages needed, notably `diplomacy>=1.1.2`, `openai>=0.27.0`, plus the dependencies from `welfare_diplomacy_baselines`.

---

## Getting Started

1. **Clone this repo**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DiplomacyBench.git
   cd DiplomacyBench
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This includes `python-diplomacy`, `openai`, etc.

3. **Download RL policy weights**:
   We rely on the [**no-press policy** from the welfare_diplomacy_baselines project](https://github.com/hannaherlebach/welfare_diplomacy_baselines).  
   - Download the weights file from:  
     https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz  
   - Place it in:
     ```bash
     welfare_diplomacy_baselines/network_parameters/fppi2_params.npz
     ```

4. **Set up your LLM credentials**:
   - Copy `.env.example` to `.env` (or just use `.env` if provided) and edit `OPENAI_API_KEY`, `OPENAI_BASE_URL`, etc.  
   - Set `TEST_AGENT_MODEL` -- this is the model whose performance will be benchmarked.
   - Update `DEFAULT_AGENT_MODEL` if you want to use a different base model.

5. **Run a test game**:
   ```bash
   python main.py --game-id=demo_game --turns=5 --negotiate
   ```
   - `--negotiate` activates full-press negotiations (sub-rounds).  
   - `--game-id` is the save/load identifier. A JSON log file named `demo_game.json` will be created.  
   - `--turns=5` runs up to 5 game years.

6. **Review game logs**:
   - The moves, negotiations, and final game state are saved in `demo_game.json`.
   - Turn-by-turn rendered `.svg` maps get stored in `gamestate_renders/demo_game/`.

7. **Generate a report**:
   - Open `generate-reports.ipynb` and run to produce tables summarizing who owned which supply centers, who allied with whom, etc.

8. **Create an animation**:
   - Open `generate-webp.ipynb` to create an animated `.webp` visualization of the game’s progression.

---

## Code Flow Summary

- **`main.py`** orchestrates a multi-power game:
  1. Creates a `DiplomacyEnvironment`.
  2. Creates one `LLMAgent` per power, each with unique personality text.
  3. Enters a loop (up to `N` turns):
     - For movement phases, if `--negotiate` is set, runs multiple sub-rounds of agent negotiations via `compose_missives()`.  
     - Each agent calls the RL policy plus its LLM logic to decide final orders.  
     - The environment processes these orders using the python-diplomacy engine.  
     - Results (supply center changes, dislodged units, etc.) are logged.
  4. The game ends upon a standard Diplomacy conclusion or after the max turn limit.

- Each **LLMAgent**:
  - Maintains a private “journal” capturing internal reasoning and relevant events.
  - Receives RL “suggestions” but can follow or ignore them.
  - Exchanges short text missives (with potential deception or alliance formation).
  - Issues final orders in JSON form, which the engine validates and adjudicates.

---

## Challenges & Approaches

1. **Rich, complex game state**  
   Diplomacy’s board has many provinces, possible supports, convoy chains, alliances, etc. We encode the board state and adjacency data in text that is (hopefully) parseable by the LLM, but still concise enough not to exceed token limits.
2. **Negotiations**  
   Full-press means large freeform text. We structure repeated sub-round calls to the LLM where it composes or receives short “missives.”  
3. **Zero-shot LLM usage**  
   Agents are told to produce valid JSON for orders, as well as short text for negotiations. We filter or correct obvious errors in code.

---

## Credits

The codebase uses:
1. The [python-diplomacy](https://github.com/boardgamers/diplomacy) package for the **Diplomacy** game engine (adjudication, movement, supply center tracking, etc.).
2. A **no-press reinforcement learning policy** for recommended moves, borrowed and adapted from [welfare_diplomacy_baselines](https://github.com/hannaherlebach/welfare_diplomacy_baselines). This policy acts as a strategic hint to the LLM (only provided during the negotiation phase), which it may or may not follow. The main purpose of the RL policy's move suggestions is to help the baseline AIs to play more competitively, and avoid the otherwise common stalemates. This can be disabled if you prefer to see the LLMs play with only their natural wits.

- **Engine**: [python-diplomacy](https://github.com/boardgamers/diplomacy)  
- **RL policy**: Adapted from [welfare_diplomacy_baselines](https://github.com/hannaherlebach/welfare_diplomacy_baselines).
- Research references:
  ```bibtex
  @misc{anthony2020learning,
    title={Learning to Play No-Press Diplomacy with Best Response Policy Iteration},
    author={Thomas Anthony and Tom Eccles and Andrea Tacchetti and János Kramár
            and Ian Gemp and Thomas C. Hudson and Nicolas Porcel
            and Marc Lanctot and Julien Pérolat and Richard Everett
            and Roman Werpachowski and Satinder Singh and Thore Graepel
            and Yoram Bachrach},
    year={2020},
    eprint={2006.04635},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
  ```

Feedback, suggestions, or pull requests are very welcome!
