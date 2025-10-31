# File: ai/diplo/diplogen-ai/diplomacy_game/persistence.py
import json
import os
import logging
from gamescope.environments.diplomacy.vendor.diplobench.diplomacy_game.environment import DiplomacyEnvironment
from gamescope.environments.diplomacy.vendor.diplobench.diplomacy_game.agent import LLMAgent

logger = logging.getLogger(__name__)

def save_game_state(game_id, env, agents, issued_orders=None, accepted_orders=None):
    """
    Saves environment and agent states to a JSON file: <game_id>.json.
    Also appends the current phase's game state, orders issued, and orders accepted
    to a turn_history list so that every turn is recorded in the same file.
    """
    if issued_orders is None:
        issued_orders = {}
    if accepted_orders is None:
        accepted_orders = {}

    game_state = env.get_game_state_dict()
    agents_data = {}
    for power, agent in agents.items():
        agents_data[power] = {
            "personality": agent.personality,
            "goals": agent.goals,
            "journal": agent.journal,
            "model_name": agent.model_name,
            "relationships": agent.relationships
        }

    # Prepare the current turn record with structured data.
    current_turn_record = {
        "phase": env.game.get_current_phase(),
        "env_state": env.game.get_state(),
        "issued_orders": issued_orders,  # Orders given by the LLM
        "accepted_orders": accepted_orders,  # Orders accepted by the game engine
        "player_relationships": {pwr: agent.relationships for pwr, agent in agents.items()}
    }

    fname = f"{game_id}.json"
    # If the file exists, load its content to retrieve previous turn_history
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading existing game state from {fname}: {e}")
            old_data = {}
    else:
        old_data = {}

    # Retrieve existing turn_history if present; otherwise, start a new list.
    turn_history = old_data.get("turn_history", [])
    turn_history.append(current_turn_record)

    # Build the save dictionary.
    save_dict = {
        "game_id": game_id,
        "env_state": game_state,
        "done": env.done,
        "agents_data": agents_data,
        "turn_history": turn_history,
        "negotiation_history": env.negotiation_history if hasattr(env, "negotiation_history") else []
    }

    with open(fname, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)
    logger.info(f"Game state saved to {fname}")



def load_game_state(game_id):
    fname = f"{game_id}.json"
    if not os.path.isfile(fname):
        return None, None

    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loading game state from {fname}...")

        env = DiplomacyEnvironment()
        env.game.set_state(data["env_state"])
        env.done = data.get("done", False)

        # Load negotiation history
        env.negotiation_history = data.get("negotiation_history", [])

        # Load agents
        agents = {}
        for power, adata in data["agents_data"].items():
            ag = LLMAgent(
                power_name=power,
                personality=adata["personality"],
                goals=adata["goals"],
                journal=adata["journal"],
                model_name=adata["model_name"]
            )
            # restore relationships if present
            if "relationships" in adata:
                ag.relationships = adata["relationships"]
            agents[power] = ag

        return env, agents
    except Exception as e:
        logger.error(f"Error loading game from {fname}: {e}")
        return None, None
