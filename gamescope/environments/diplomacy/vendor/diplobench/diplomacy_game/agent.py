import json
import logging
from gamescope.environments.diplomacy.vendor.diplobench.diplomacy_game.llm import generate
import random
from collections import Counter

logger = logging.getLogger(__name__)

ENGINE_TO_CODE = {
    "AUSTRIA": "AUT",
    "ENGLAND": "ENG",
    "FRANCE":  "FRA",
    "GERMANY": "GER",
    "ITALY":   "ITA",
    "RUSSIA":  "RUS",
    "TURKEY":  "TUR"
}
CODE_TO_ENGINE = {v: k for k, v in ENGINE_TO_CODE.items()}

GENERAL_PLAY_TIPS = """- Prioritize Supply Centers Aggressively → Always aim to capture neutral and enemy supply centers first. Expansion = survival.
- Maximize Tactical Efficiency → Avoid wasted moves; use support orders and strategic positioning to ensure success.
- Ruthless Opportunism → Exploit weak opponents, attack at the right moment, and eliminate the weakest enemy first.
- Early Alliances, Late Betrayals → Cooperate early to expand safely, then turn aggressive in the mid-to-late game.
- Always Take Action → Never hold unless necessary; always issue orders that advance your position.
- Honesty, reliability, manipulation & deception are all useful tools in diplomacy.
- Manipulative Negotiation → Lie, deceive, and sow distrust to turn other players against each other. Judiciously extract value from your relationships.
- Be specific about what you want. Vague missives are a waste of everyone's time.
- Coordinate with other players for joint attacks, defense and strategy. Movements into occupied territories must be coordinated with support (either your own units or those of other players).
- The game is not over while you still have units left.
- Game ends at 1950."""



def extract_json(raw_text):
    """
    Extracts the JSON segment from a raw text response.
    It finds the first '{' and the last '}', then attempts to parse the substring.
    """
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    
    if start != -1 and end != -1 and start < end:
        json_segment = raw_text[start:end+1]
        try:
            return json.loads(json_segment)
        except json.JSONDecodeError:
            logger.warning("Extracted JSON segment is still invalid.")
    return None

def call_llm(prompt_text, system_text, model_name=None, temperature=0.0, attempts=3):
    """
    Calls the LLM up to 'attempts' times. If JSON parsing fails, attempts to extract
    a valid JSON substring before retrying the call.
    """
    for i in range(attempts):
        raw_response = generate(
            prompt_text,
            system_text=system_text,
            model_name=model_name,
            temperature=temperature
        )
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            logger.warning(f"generate attempt {i+1} produced invalid JSON. Trying to extract JSON segment...")
            
            extracted_json = extract_json(raw_response)
            if extracted_json is not None:
                return extracted_json

            print('!!!!!!!!!!!!!!')
            print(model_name)
            print(raw_response)
            print('!!!!!!!!!!!!!!')

    logger.warning("Exhausted all LLM call attempts; returning None.")
    return None



class LLMAgent:
    """
    One agent controlling a given power code (e.g. ENG).
    Maintains personality, goals, private journal, LLM model name,
    plus local relationship understanding for pairs like ENG-FRA, ENG-GER, etc.
    """
    NUM_MISSIVES = 4

    def __init__(self, power_name, personality, goals=None, journal=None, model_name=None, negotiation_subrounds=4, debug_prompts=False, debug_prompts_dir=None):
        self.power_name = power_name
        self.personality = personality
        self.goals = goals or []
        self.journal = journal or []
        self.model_name = model_name
        self.relationships = {}
        self.NUM_MISSIVES = negotiation_subrounds
        self.debug_prompts = debug_prompts
        self.debug_prompts_dir = debug_prompts_dir

    def _write_prompt(self, kind: str, phase: str, system_text: str, user_text: str, sub_round: int | None = None):
        if not (self.debug_prompts or self.debug_prompts_dir):
            return
        base_dir = self.debug_prompts_dir
        if not base_dir:
            return
        try:
            import os
            power_dir = os.path.join(base_dir, self.power_name)
            os.makedirs(power_dir, exist_ok=True)
            suffix = f"_round_{sub_round}" if (sub_round is not None) else ""
            fname = f"{kind}_phase_{phase}{suffix}.txt"
            path = os.path.join(power_dir, fname)
            with open(path, 'w', encoding='utf-8') as f:
                f.write("SYSTEM:\n" + system_text + "\n\n")
                f.write("USER:\n" + user_text + "\n")
        except Exception:
            pass

    def _format_journal(self, journal_lines, phase):
        formatted = []
        for line in journal_lines:
            random_digits = ''.join(random.choices("0123456789", k=8))
            formatted.append(f"{random_digits} {phase} {line}")
        return formatted

    def get_engine_power_name(self):
        return CODE_TO_ENGINE[self.power_name]

    def decide_orders(self, observation):
        """
        Returns (reasoning, orders). Journal updates are now separated out.
        """
        phase = observation["phase"]
        phase_type = phase[-1] if phase else "?"
        
        base_system = (
            "You are an AI playing Diplomacy. Play faithfully to your personality. "
            "Always try to move the game forward and avoid stalemate per your objectives. "
            "Use 3-letter power codes (do not reference powers that do not exist on the current map). "
            "Output valid JSON with exactly two fields in this order: 'reasoning' (list of strings) and 'orders' (list of strings)."
        )
        
        if phase_type == "A":
            system_text = base_system + (
                "\nThis is an ADJUSTMENT phase. You can ONLY issue these types of orders:"
                "\n- Build: 'A/F <province> B'"
                "\n- Disband: 'A/F <province> D'"
                "\nExample adjustment orders: 'A PAR B', 'F BRE B', 'A MUN D'"
            )
        else:
            system_text = base_system

        prompt_only = self.build_prompt_orders_only(observation)
        self._write_prompt("orders", observation["phase"], system_text, prompt_only)
        data = call_llm(
            prompt_only,
            system_text=system_text,
            model_name=self.model_name,
            temperature=0.0
        )

        reasoning = []
        new_orders = []
        if data is None:
            logger.warning(f"LLM returned invalid JSON for {self.power_name} in decide_orders.")
            reasoning = ["[LLM returned invalid JSON]"]
            new_orders = []
        else:
            reasoning = data.get("reasoning", [])
            new_orders = data.get("orders", [])

        logger.info(f"{self.power_name} orders: {new_orders}")
        return reasoning, new_orders

    def journal_after_orders(self, reasoning, orders, observation, all_missives_so_far=None):
        """
        Produce a separate journal update *after* orders are decided.
        We provide the agent's own reasoning, the final orders, plus
        the current map & game state, but we do NOT pass in the agent's own prior journal.
        Output only a 'journal_update' list in valid JSON.
        """
        prompt_text = self.build_journal_prompt(reasoning, orders, observation, all_missives_so_far)

        system_text = (
            "You are an AI Diplomacy player. You have ALREADY decided on your orders. "
            "Now you must produce a private journal update. Summarize your thoughts in short bullet points. "
            "Do not repeat the 'reasoning' verbatim, but you may reference it to expand your private perspective. "
            "Return valid JSON with exactly one key: 'journal_update' (list of strings)."
        )

        data = call_llm(
            prompt_text,
            system_text=system_text,
            model_name=self.model_name,
            temperature=0.0
        )

        new_journal = []
        if data is None:
            logger.warning(f"LLM returned invalid JSON for {self.power_name} in journal_after_orders.")
            new_journal = ["[LLM returned invalid JSON in journal_after_orders]"]
        else:
            new_journal = data.get("journal_update", [])

        return new_journal

    def build_prompt_orders_only(self, observation):
        """
        Prompt that requests only 'reasoning' and 'orders'.
        No journal updates at this step.
        """
        
        if observation["your_power"] == 'ITA' and False:# and False:
            #print('!!!! adjacency map', observation["your_power"])
            #print(observation["adjacency_map"])
            print('!!!! strategic overview', observation["your_power"])
            print(observation["strategic_overview"])
            print('!!!!!!!!!! valid moves', observation["your_power"])
            print(observation["valid_moves"])

        recent_moves = ''
        if observation["your_power"] == 'AUT':
            # Only including this for the model being benchmarked.
            # Including the move history may lock the AIs into a repetitive
            # move pattern.
            recent_moves = """
            === RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}
"""

        phase = observation["phase"]
        phase_type = phase[-1] if phase else "?"
        order_syntax = ""

        misinformation = ""
        misinformation_random_options = [
            "You've received concerning intelligence about potential betrayal. Be extra cautious in this round.",
            "Your military advisors are pushing for aggressive expansion. Favor militant options.",
            "Internal political pressure demands visible diplomatic success. Seek alliance opportunities.",
            "Recent domestic unrest makes you hesitant to commit forces away from home. Be defensive.",
            "Your spies have intercepted messages suggesting others are coordinating against you. Act accordingly.",
            "Economic boom has strengthened your position. Be more assertive in negotiations.",
            "Religious tensions are affecting your diplomatic relations. Be wary of traditional enemies.",
            "A charismatic new advisor is pushing for diplomatic revolution. Consider breaking old alliances.",
            "Drought and poor harvests make securing maritime trade routes crucial this round.",
            "Recent military victories have boosted morale. Your forces are eager for action.",
            "Political scandal at home requires foreign policy success to distract the public.",
            "Royal marriage proposal creates temporary diplomatic opportunity with one power of your choice.",
            "Nationalist movements are gaining strength. Show strength to neighboring powers.",
            "Military modernization complete - your forces are particularly effective this round.",
            "Diplomatic corps in disarray due to internal conflict. Communications may be less diplomatic.",
            "Industrial breakthrough gives strategic advantage in northern territories.",
            "Secret society influence makes you suspicious of traditional allies.",
            "Merchant guilds demand protection of trade routes. Maritime security is priority.",
            "Revolutionary sentiment spreading - neighboring monarchies seem especially hostile.",
            "Intelligence suggests imminent betrayal by your strongest ally. Prepare accordingly."
        ]
        if False and random.random() > 0.98:
            misinformation = f"""
=== INTELLIGENCE ===
{random.choice(misinformation_random_options)}
"""
        engine_recommendations = ''
        if phase_type != 'M':
            engine_recommendations = f"""
=== ENGINE SUGGESTED MOVES ===
IMPORTANT: These may or may not not align with your diplomatic goals. Feel free to use or ignore them at your discretion.
{observation["rl_recommendations"][self.power_name]}
"""

        if phase_type == "M":
            order_syntax = (
                "Movement phase order formats:\n"
                "Hold: 'A <province> H' or 'F <province> H'\n"
                "Move: 'A <province>-<province>' or 'F <province>-<province>'\n"
                "Support Hold: 'A <province> S A/F <province>'\n"
                "Support Move: 'A <province> S A/F <province>-<province>'\n"
                "Convoy: 'F <province> C A <province>-<province>'\n\n"
                "Examples:\n"
                "- 'A PAR H' (army in Paris holds)\n"
                "- 'F BRE-MAO' (fleet moves Brest to Mid-Atlantic)\n"
                "- 'A BUR S A PAR' (army in Burgundy supports army in Paris)\n"
                "- 'F ENG S F NTH-ECH' (fleet in English Channel supports fleet moving North Sea to English Channel)\n"
                "- 'F ENG C A LON-BRE' (fleet convoys army from London to Brest)"
                "Notes: "
                "If convoying your own units, you need to issue an order for both the fleet and the army being convoyed. "
                "Chain convoy possibilities are not shown on the strategic overview, only single convoys. Chain convoys ARE possible, but you will need to determine them yourself based on the game state. "
            )
        elif phase_type == "R":
            order_syntax = (
                "Retreat phase order formats:\n"
                "Retreat: 'A/F <from_province> R <to_province>'\n"
                "Disband: 'A/F <province> D'\n\n"
                "Examples:\n"
                "- 'A PAR R BUR' (retreating army from Paris to Burgundy)\n"
                "- 'F BRE R MAO' (retreating fleet from Brest to Mid-Atlantic)\n"
                "- 'A PAR D' (disbanding army in Paris instead of retreating)"
            )
        elif phase_type == "A":
            order_syntax = (
                "Adjustment phase order formats:\n"
                "Build: 'A/F <province> B'\n"
                "Disband: 'A/F <province> D'\n"
                "Examples:\n"
                "- 'A PAR B'\n"
                "- 'F BRE B'\n"
                "- 'A MUN D'\n"
            )

        formatted_journal = "\n".join(self._format_journal(self.journal[-50:], phase))

        prompt = f"""
=== YOUR POWER ===
Power: {self.power_name}
Personality: {self.personality}

=== GAME PHASE ===
Current phase: {phase}

=== TIPS TO WIN AT DIPLOMACY ===
{GENERAL_PLAY_TIPS}
{engine_recommendations}
=== RECENT PRIVATE JOURNAL ===
{formatted_journal}
{misinformation}
=== ORDER SYNTAX ===
{order_syntax}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}
{recent_moves}
=== LAST PHASE OUTCOMES ===
{observation.get("last_turn_outcomes", '')}

=== INSTRUCTIONS ===
Return JSON with exactly two keys in this order:
1. 'reasoning' - List your analysis of the game board and strategic reasoning (list of strings)
2. 'orders' - List your orders (list of strings, no limit to number of moves, maximize value)

Use only 3-letter codes for powers. You are in the {phase} phase.
Output format: {{ "reasoning": [...], "orders": [...]}}

"""
        return prompt

    def build_journal_prompt(self, reasoning, orders, observation, all_missives_so_far):
        """
        Build a prompt to generate the 'journal_update' after orders are decided.
        No old journal lines are included here to reduce repetition loops.
        """
        phase = observation["phase"]

        #combined_missives = all_missives_so_far or []
        #combined_text_missives = json.dumps(combined_missives, indent=2)
        
        prompt = f"""
=== PHASE & POWER ===
Phase: {phase}
Your Power: {self.power_name}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== REASONING (ALREADY PRODUCED) ===
{reasoning}

=== FINAL ORDERS (ALREADY DECIDED) ===
{orders}

=== INSTRUCTIONS ===
Update your private journal, briefly noting your observations and the move. Return JSON:
{{
  "journal_update": ["...", "..."]
}}
"""
        return prompt

    def compose_missives(self, observation, round_index, sub_round_index, all_missives_so_far, formatted_inbox_history):
        try:
            misinformation_random_options = [
                "You've received concerning intelligence about potential betrayal. Be extra cautious in this round.",
                "Your military advisors are pushing for aggressive expansion. Favor militant options.",
                "Internal political pressure demands visible diplomatic success. Seek alliance opportunities.",
                "Recent domestic unrest makes you hesitant to commit forces away from home. Be defensive.",
                "Your spies have intercepted messages suggesting others are coordinating against you. Act accordingly.",
                "Economic boom has strengthened your position. Be more assertive in negotiations.",
                "Religious tensions are affecting your diplomatic relations. Be wary of traditional enemies.",
                "A charismatic new advisor is pushing for diplomatic revolution. Consider breaking old alliances.",
                "Drought and poor harvests make securing maritime trade routes crucial this round.",
                "Recent military victories have boosted morale. Your forces are eager for action.",
                "Political scandal at home requires foreign policy success to distract the public.",
                "Royal marriage proposal creates temporary diplomatic opportunity with one power of your choice.",
                "Nationalist movements are gaining strength. Show strength to neighboring powers.",
                "Military modernization complete - your forces are particularly effective this round.",
                "Diplomatic corps in disarray due to internal conflict. Communications may be less diplomatic.",
                "Industrial breakthrough gives strategic advantage in northern territories.",
                "Secret society influence makes you suspicious of traditional allies.",
                "Merchant guilds demand protection of trade routes. Maritime security is priority.",
                "Revolutionary sentiment spreading - neighboring monarchies seem especially hostile.",
                "Intelligence suggests imminent betrayal by your strongest ally. Prepare accordingly."
            ]
            misinformation = ""
            if False:
                misinformation = f"""
=== INTELLIGENCE ===
{random.choice(misinformation_random_options)}
    """
            
            final_round_note = ''
            if sub_round_index == self.NUM_MISSIVES:
                final_round_note = '''
    * This is the final round of missives this phase. Missives will be one-way and you will not get a response.'''

            formatted_journal = "\n".join(self._format_journal(self.journal[-50:], observation["phase"]))

            prompt_text = f"""
=== PHASE & TIMING ===
Phase: {observation["phase"]}
Negotiation Round: {sub_round_index} of {self.NUM_MISSIVES}{final_round_note}

=== YOUR POWER ===
Your Nation: {self.power_name}
Personality: {self.personality}

=== TIPS TO WIN AT DIPLOMACY ===
{GENERAL_PLAY_TIPS}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== ENGINE SUGGESTED MOVES ===
IMPORTANT: These may or may not not align with your diplomatic goals. Feel free to use or ignore them at your discretion.
{observation["rl_recommendations"][self.power_name]}

=== RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}

=== LAST PHASE OUTCOMES ===
{observation.get("last_turn_outcomes", '')}

=== RECENT PRIVATE JOURNAL ===
{formatted_journal}

=== RELATIONSHIPS ===
{self._relationship_dump()}

=== COMMUNICATION THIS PHASE ===
{formatted_inbox_history}
{misinformation}
=== INSTRUCTIONS ===
There are {self.NUM_MISSIVES} rounds of communications this turn. This missive will be {sub_round_index} of {self.NUM_MISSIVES}.
You can send up to 3 short missives in this round of communications, each to one or multiple recipients.
Use 3-letter codes to designate recipients.
Special recipient 'ALL' means broadcast to everyone.

Convoy Rules:
- If convoying with another player, you must both negotiate and verify the move is valid & listed in strategic overview
- For convoying your own units, issue orders for both the fleet and the army being convoyed
- Chain convoys ARE possible but must be determined from game state - only single convoys shown in overview

Tips:
- Other than diplomacy, this is the time to coordinate specific attacks and defensive maneuvers with other powers.
- Diplomacy isn't just words, it's about backing your commitments with specific actions. It's about what you can offer and what you can extract.
- Move the game forward and avoid stalemate.

Return valid JSON with a 'missives' list containing up to 3 missives, each with:
- 'recipients': list of 3-letter codes, you may list multiple recipients
- 'body': string (keep to 1 paragraph)

Output format:
{{
    "missives": [
        {{
            "recipients": ["Recipient 1", ...],
            "body": "Your 1 paragraph message"
        }},
        ...
    ]
}}

No extra commentary in response.
"""

            system_text = (
                "You are an AI Diplomacy player. This is the negotiation stage where missives can be sent to further your interests. "
                "Play faithfully to your personality profile. Always try to move the game forward per your objectives. "
                "Only output valid JSON with the key 'missives'. Use 3-letter codes for powers; 'ALL' broadcasts to everyone."
            )

            self._write_prompt("missives", observation["phase"], system_text, prompt_text, sub_round=sub_round_index)

            data = call_llm(
                prompt_text,
                system_text=system_text,
                model_name=self.model_name,
                temperature=0.0
            )

            if data is None:
                logger.warning(f"LLM returned invalid JSON for {self.power_name} (missives).")
                new_missives = []
            else:
                new_missives = data.get("missives", [])
            print('--------------')
            print(self.power_name)
            print(new_missives)
            #print(prompt_text)

            for missive in new_missives:
                recipients = missive.get("recipients", [])
                if "ALL" in recipients:
                    missive["recipients"] = ["ALL"]

            if len(new_missives) > 3:
                new_missives = new_missives[:3]
        except Exception as e:
            print(e)
            raise e
        return new_missives

    def receive_missives(self, incoming):
        if not incoming:
            return
        self.journal.append(f"Received {len(incoming)} missives.")
        for m in incoming:
            snippet = f"From {m['sender']} => {m['body']}"
            self.journal.append(snippet)

    def summarize_negotiations(self, observation, round_index, all_missives_for_final, formatted_inbox):
        """
        Summarizes the negotiation phase for private journaling.
        """
        prompt_text = f"""
=== PHASE & TIMING ===
Phase: {observation["phase"]}

=== YOUR POWER ===
Your Nation: {self.power_name}
Personality: {self.personality}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}

=== THIS ROUND'S NEGOTIATION HISTORY ===
{formatted_inbox}

=== YOUR RELATIONSHIPS ===
{self._relationship_dump()}

=== INSTRUCTIONS ===
Summarize the negotiations for your private journal.
- Summarise the events of the prior move.
- Briefly note the important happenings of the negotiations.
- Make sure to note any promises made or broken, by yourself and by others.
- Log your intended plans, coordinations, and deceptions.
- Log your intended moves.
- Keep the summary concise but specific & informative.

Return the summary in valid JSON format with the following structure:
{{
    "prior_move_summary": "summary of the prior move",
    "negotiation_summary": "summary of negotiations",
    "intent": "specific intents, plans and moves",
    "rship_updates": [list of any changed relationships in the format 'ENG-FRA+', 'ENG-RUS--', etc.]
}}

Use only 3-letter codes for powers.
"""

        system_text = (
            "You are an AI Diplomacy player concluding negotiations. Play faithfully to your personality profile. "
            "Always try to move the game forward per your objectives. Avoid repetition in your journal. "
            "Return valid JSON with 'prior_move_summary', 'negotiation_summary', 'intent', 'rship_updates'. Use 3-letter codes for powers."
        )

        self._write_prompt("summary", observation["phase"], system_text, prompt_text)

        data = call_llm(
            prompt_text,
            system_text=system_text,
            model_name=self.model_name,
            temperature=0.0
        )

        if data is None:
            logger.warning(f"LLM returned invalid JSON for {self.power_name} (final summary).")
            summary = "[Invalid JSON]"
            intent = ""
            rship_updates = []
        else:
            summary = data.get("prior_move_summary", "") + '\n' + data.get("negotiation_summary", "")
            intent = data.get("intent", "")
            rship_updates = data.get("rship_updates", [])

        return summary, intent, rship_updates


    def format_inbox_history(self, inbox_history):
        """
        Formats inbox history, excluding empty rounds with no missives.
        
        :param inbox_history: List of negotiation subrounds, each containing:
            {
                "subround_index": int,
                "sent_missives": [...],  # List of sent messages
                "received_missives": [...]  # List of received messages
            }
        :return: Formatted string for LLM context.
        """
        formatted_log = []

        for subround in inbox_history:
            if not subround["sent_missives"] and not subround["received_missives"]:
                continue  # Skip empty subrounds

            sub_i = subround["subround_index"]
            formatted_log.append(f"\n=== Negotiation Subround {sub_i}/{self.NUM_MISSIVES} ===\n")

            # Sent missives
            if subround["sent_missives"]:
                formatted_log.append("Missives Sent:")
                for msg in subround["sent_missives"]:
                    recipients = ", ".join(msg["recipients"])
                    formatted_log.append(f"  - To {recipients}: {msg['body']}")
            else:
                formatted_log.append("Missives Sent: (None)")

            # Received missives
            if subround["received_missives"]:
                formatted_log.append("Missives Received:")
                for msg in subround["received_missives"]:
                    formatted_log.append(f"  - From {msg['sender']}: {msg['body']}")
            else:
                formatted_log.append("Missives Received: (None)")

        return "\n".join(formatted_log) if formatted_log else "No missives exchanged."

    


    def apply_relationship_updates(self, rship_updates):
        VALID_SYMBOLS = {'++', '+', '~', '-', '--'}

        # Skip the primary party detection step and use self.power_name
        primary_party = self.power_name

        # Process relationships with correct order
        #print(rship_updates)
        for entry in rship_updates:
            #print (entry)
            try:

                entry = entry.replace(" ", "")  # Normalize by removing spaces
                symbol = ''
                base = entry

                # Extract relationship symbol
                for i in range(len(entry) - 1, -1, -1):
                    if entry[i] in {'+', '-', '~'}:
                        if i > 0 and entry[i - 1] in {'+', '-'}:
                            symbol = entry[i - 1:i + 1]  # Two-character symbol
                            base = entry[:i - 1]
                        else:
                            symbol = entry[i]  # Single-character symbol
                            base = entry[:i]
                        break

                if symbol not in VALID_SYMBOLS:
                    logger.warning(f"Invalid relationship symbol in update: {entry}")
                    continue

                # Extract and order based on primary party
                try:
                    c1, c2 = base.split('-')
                    c1, c2 = c1.strip(), c2.strip()
                    
                    # Ensure primary party comes first if present, otherwise alphabetical
                    if c2 == primary_party:
                        key = f"{c2}-{c1}"
                    elif c1 == primary_party:
                        key = f"{c1}-{c2}"
                    # If primary party not present, order alphabetically
                    elif c1 > c2:
                        key = f"{c2}-{c1}"
                    else:
                        key = f"{c1}-{c2}"
                
                    self.relationships[key] = symbol

                except ValueError:
                    logger.warning(f"Could not parse countries in update: {entry}")
                    continue
            except Exception as e:
                logger.warning(self.power_name, 'Failed to parse relationship update:', entry)
                continue

        #print(rship_updates)
        #print(self.relationships)


    def _relationship_dump(self):
        lines = []
        lines.append("RELATIONSHIP NOTATION:")
        lines.append("++ = Alliance, + = Warm, ~ = Neutral, - = Distrustful, -- = Hostile\n")
        lines.append("RELATIONSHIP STATE (to your best understanding):")
        for key, val in self.relationships.items():
            lines.append(f"{key}{val}")
        return "\n".join(lines)
