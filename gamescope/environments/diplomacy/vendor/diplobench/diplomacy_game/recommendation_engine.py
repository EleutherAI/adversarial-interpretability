# File: ai/diplo/diplogen-ai/diplomacy_game/recommendation_engine.py

import logging
from gamescope.environments.diplomacy.vendor.diplobench.welfare_diplomacy_baselines.baselines import no_press_policies
from gamescope.environments.diplomacy.vendor.diplobench.welfare_diplomacy_baselines.environment import diplomacy_state
from gamescope.environments.diplomacy.vendor.diplobench.welfare_diplomacy_baselines.environment import mila_actions

logger = logging.getLogger(__name__)

# Mapping between 3-letter codes and full power names
CODE_TO_FULL_NAME = {
    "AUT": "AUSTRIA",
    "ENG": "ENGLAND",
    "FRA": "FRANCE",
    "GER": "GERMANY",
    "ITA": "ITALY",
    "RUS": "RUSSIA",
    "TUR": "TURKEY"
}

class RecommendationEngine:
    """Provides RL policy recommendations."""

    def __init__(self):
        self.rl_policy = no_press_policies.get_network_policy_instance()

    def get_recommendations(self, game, power_code):
        """Gets action recommendations from the pre-trained RL policy."""
        try:
            # Convert 3-letter code to full name
            power_name = CODE_TO_FULL_NAME.get(power_code)
            if not power_name:
                logger.warning(f"Unknown power code: {power_code}")
                return []
                
            state = diplomacy_state.WelfareDiplomacyState(game)
            power_idx = game.map.powers.index(power_name)
            observation = state.observation()
            legal_actions = state.legal_actions()

            self.rl_policy.reset()
            actions, _ = self.rl_policy.actions([power_idx], observation, legal_actions)
            actions = actions[0]

            orders = []
            for action in actions:
                candidate_orders = mila_actions.action_to_mila_actions(action)
                order = mila_actions.resolve_mila_orders(candidate_orders, game)
                orders.append(order)
            return orders

        except Exception as e:
            logger.error(f"Error getting RL recommendations for {power_code}: {e}")
            return []