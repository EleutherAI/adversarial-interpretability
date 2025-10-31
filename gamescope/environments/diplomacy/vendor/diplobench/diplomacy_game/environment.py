from diplomacy import Game
import logging
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Mapping between python-diplomacy default power names and our 3-letter codes.
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

def _build_power_code_mapping(engine_power_names):
    """Build a stable mapping from engine power names to short codes.

    - Prefer existing well-known mappings when names match the standard map.
    - Otherwise generate 3-letter uppercase codes from the name.
    - Ensure uniqueness by appending numeric suffixes when necessary.
    """
    standard = {
        "AUSTRIA": "AUT",
        "ENGLAND": "ENG",
        "FRANCE": "FRA",
        "GERMANY": "GER",
        "ITALY": "ITA",
        "RUSSIA": "RUS",
        "TURKEY": "TUR",
    }

    proposed = {}
    used = set()

    for name in engine_power_names:
        up = name.upper()
        base = None
        if up in standard:
            base = standard[up]
        else:
            letters = ''.join([c for c in up if c.isalpha()])
            base = (letters[:3] if len(letters) >= 3 else (letters + 'XXX')[:3])
        code = base
        idx = 2
        while code in used:
            # Keep 2 leading letters and append a digit to resolve collisions
            code = (base[:2] + str(idx))[:3]
            idx += 1
        proposed[up] = code
        used.add(code)
    return proposed

def _engine_state_to_code_state(engine_state):
    """
    Convert a game.get_state() dictionary that may look like:
      {
        'centers': {'AUSTRIA': [...], 'FRANCE': [...]},
        'units':   {'ENGLAND': [...], ...},
        ...
      }
    into a version that uses codes for power keys:
      {
        'centers': {'AUT': [...], 'FRA': [...]},
        'units':   {'ENG': [...], ...},
        ...
      }
    We do a shallow transform of known keys: 'centers', 'units', 'orders', 'retreats', 'adjustments', etc.
    """
    if not engine_state:
        return {}

    code_state = {}
    for k, v in engine_state.items():
        if isinstance(v, dict):
            converted_dict = {}
            for eng_power, val in v.items():
                code_power = ENGINE_TO_CODE.get(eng_power, eng_power)
                converted_dict[code_power] = val
            code_state[k] = converted_dict
        else:
            code_state[k] = v

    return code_state

from collections import deque



class DiplomacyGraph:
    """Custom graph implementation for Diplomacy map connectivity."""
    
    def __init__(self):
        # Main graph structure: dict of dict of sets
        # graph[node1][node2] = {'A', 'F'} means both army and fleet can move between nodes
        # graph[node1][node2] = {'A'} means only army can move between nodes
        self.graph = {}
        
    def add_node(self, node):
        """Add a node if it doesn't exist."""
        if node not in self.graph:
            self.graph[node] = {}
            
    def add_edge(self, node1, node2, unit_type):
        """Add an edge between nodes for specific unit type ('A' or 'F')."""
        self.add_node(node1)
        self.add_node(node2)
        
        # Add connection for node1 -> node2
        if node2 not in self.graph[node1]:
            self.graph[node1][node2] = set()
        self.graph[node1][node2].add(unit_type)
        
        # Add connection for node2 -> node1 (undirected graph)
        if node1 not in self.graph[node2]:
            self.graph[node2][node1] = set()
        self.graph[node2][node1].add(unit_type)
        
    def get_adjacent(self, node):
        """Get all nodes adjacent to given node."""
        return self.graph[node].keys()
        
    def get_allowed_units(self, node1, node2):
        """Get set of unit types that can move between these nodes."""
        return self.graph[node1][node2]
        
    def nodes(self):
        """Return all nodes in the graph."""
        return self.graph.keys()
        
    def edges(self):
        """Return all edges with their unit types as (node1, node2, unit_types)."""
        edges = []
        seen = set()  # To avoid duplicates in undirected graph
        
        for node1 in self.graph:
            for node2, unit_types in self.graph[node1].items():
                if (node1, node2) not in seen and (node2, node1) not in seen:
                    edges.append((node1, node2, unit_types))
                    seen.add((node1, node2))
                    
        return edges

class DiplomacyEnvironment:
    """
    Wraps the python-diplomacy Game object, tracking state and execution.
    Internally, the engine uses AUSTRIA, ENGLAND, FRANCE, etc.
    Externally, we present AUT, ENG, FRA, etc.
    """

    def __init__(self, game_state_dict=None, map_name_or_path='standard'):
        """
        If game_state_dict is provided, load that state.
        Otherwise, start a fresh game with the specified map (name or full path).
        Note that the python-diplomacy library will create powers internally
        named 'AUSTRIA','ENGLAND','FRANCE','GERMANY','ITALY','RUSSIA','TURKEY'.
        """
        self.game = Game(map_name=map_name_or_path)  # explicit map_name supports file paths
        self.done = False

        # Build dynamic power-code mapping based on the loaded map
        try:
            engine_power_names = list(self.game.powers.keys())
            dynamic_map = _build_power_code_mapping(engine_power_names)
            ENGINE_TO_CODE.clear(); ENGINE_TO_CODE.update(dynamic_map)
            CODE_TO_ENGINE.clear(); CODE_TO_ENGINE.update({v: k for k, v in ENGINE_TO_CODE.items()})
        except Exception:
            # Fall back to defaults if anything goes wrong; better to crash later with clearer context
            pass

        # Seed initial units if the map starts empty (common in some variants)
        try:
            state = self.game.get_state()
            if state and all(not units for units in state.get('units', {}).values()):
                centers = state.get('centers', {})
                new_units = {}
                water_types = {'WATER', 'PORT'}
                # Precompute water neighbors for coasts
                coast_with_water = set()
                for loc in self.game.map.locs:
                    if '/' in loc:  # coast token
                        for nbr in self.game.map.abut_list(loc):
                            if self.game.map.area_type(nbr) == 'WATER':
                                coast_with_water.add(loc)
                                break
                for eng_name, sc_list in centers.items():
                    units = []
                    for sc in sc_list:
                        base = sc.split('/')[0]
                        area = self.game.map.area_type(base)
                        if area == 'LAND':
                            units.append(f"A {base}")
                        else:
                            # Try to find a coast with water; else place an army
                            candidate = None
                            for coast in self.game.map.find_coasts(base):
                                if coast in coast_with_water:
                                    candidate = coast
                                    break
                            if candidate:
                                units.append(f"F {candidate}")
                            else:
                                units.append(f"A {base}")
                    new_units[eng_name] = units
                state['units'] = new_units
                self.game.set_state(state)
        except Exception:
            pass

        if game_state_dict:
            logger.info("Loading existing game state...")
            self.game.set_state(game_state_dict)
        else:
            logger.info("Starting a new game...")
        
        self._build_connectivity_graphs()
        self.debug_connectivity()

    from collections import deque

    def bfs_shortest_path(self, start, match_condition, allowed_unit_types):
        """
        BFS from 'start' to find the first territory for which 'match_condition(territory)==True'.
        Returns (path, match) or (None, None).
        """
        if start not in self.connectivity.graph:
            return None, None
        visited = set([start])
        queue = deque([[start]])

        # If the starting territory itself satisfies match_condition
        initial_match = match_condition(start)
        if initial_match:
            return [start], initial_match

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor in self.connectivity.graph[current]:
                edge_types = self.connectivity.get_allowed_units(current, neighbor)
                if edge_types.intersection(allowed_unit_types):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        m = match_condition(neighbor)
                        if m:
                            return new_path, m
                        queue.append(new_path)
        return None, None

    def bfs_nearest_adjacent(self, start, occupant_map, allowed_unit_types):
        """
        BFS to find the first territory adjacent to a territory in occupant_map.
        occupant_map: { territory_name: occupant_unit }, for occupant units
        Returns (path, (matched_territory, occupant_unit)) or (None, (None, None)) if none found.
        """
        if not occupant_map or start not in self.connectivity.graph:
            return None, (None, None)
        visited = set([start])
        queue = deque([[start]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            # If ANY neighbor is occupant_terr => success
            for neigh in self.connectivity.graph[current]:
                if neigh in occupant_map:
                    occupant_unit = occupant_map[neigh]
                    return path, (neigh, occupant_unit)

            # else expand
            for neighbor in self.connectivity.graph[current]:
                edge_types = self.connectivity.get_allowed_units(current, neighbor)
                if edge_types.intersection(allowed_unit_types):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])

        return None, (None, None)

    def _build_connectivity_graphs(self):
        """
        Builds a connectivity graph showing all possible movements on the map by unit type.
        We'll store it in self.connectivity as a DiplomacyGraph that tracks
        which unit types ('A', 'F') can move between territories.
        """
        self.connectivity = DiplomacyGraph()
        
        # 'map.locs' is a list of all loc strings (like "PAR", "spa/sc", "STP/NC", etc.)
        all_locations = self.game.map.locs

        for loc in all_locations:
            # Convert to uppercase "base" territory name (strip off coasts, e.g. 'spa/sc' -> 'SPA')
            loc_base = loc.split('/')[0].upper()

            # Get adjacency list from the map.
            # abut_list() returns a list of neighboring loc names in "mixed" case.
            neighbors = self.game.map.abut_list(loc, incl_no_coast=False)

            for adj in neighbors:
                adj_base = adj.split('/')[0].upper()

                # Does an army have a valid move from loc to adj?
                if self.game.map.abuts('A', loc, '-', adj):
                    self.connectivity.add_edge(loc_base, adj_base, 'A')

                # Does a fleet have a valid move from loc to adj?
                if self.game.map.abuts('F', loc, '-', adj):
                    self.connectivity.add_edge(loc_base, adj_base, 'F')

        # -- (Optional) Visualization and summary stats --
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Create a NetworkX Graph with color-coded edges by unit type
        G = nx.Graph()
        for node1, node2, unit_types in self.connectivity.edges():
            if len(unit_types) == 2:   # Both Army & Fleet
                G.add_edge(node1, node2, color='green')
            elif 'A' in unit_types:    # Army-only
                G.add_edge(node1, node2, color='red')
            else:                      # Fleet-only
                G.add_edge(node1, node2, color='blue')

        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, seed=42)

        # Draw edges by color
        for color in ['red', 'blue', 'green']:
            color_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['color'] == color]
            nx.draw_networkx_edges(G, pos, edgelist=color_edges, edge_color=color)

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=8)

        legend_elements = [
            Line2D([0], [0], color='red',   label='Army Only'),
            Line2D([0], [0], color='blue',  label='Fleet Only'),
            Line2D([0], [0], color='green', label='Both'),
        ]
        plt.legend(handles=legend_elements)

        plt.savefig('connectivity_graph.png')
        plt.close()

        # Print some summary stats
        total_nodes = len(self.connectivity.nodes())
        edges_list  = list(self.connectivity.edges())
        print(f"Graph has {total_nodes} territories (nodes).")
        print(f"Total edges: {len(edges_list)}")

        army_edges  = sum(1 for _, _, types in edges_list if 'A' in types)
        fleet_edges = sum(1 for _, _, types in edges_list if 'F' in types)
        print(f"Army connections: {army_edges}")
        print(f"Fleet connections: {fleet_edges}")

    def debug_connectivity(self):
        """
        Example debug method to print out various connectivity checks,
        so we can see if the graph makes sense.
        """
        # 1) Print summary of Army-only, Fleet-only, and Both edges
        army_only    = []
        fleet_only   = []
        army_and_fleet = []
        
        for node1, node2, units in self.connectivity.edges():
            units = sorted(units)  # So we can compare easily
            if units == ['A']:
                army_only.append((node1, node2))
            elif units == ['F']:
                fleet_only.append((node1, node2))
            else:  # e.g. ['A','F']
                army_and_fleet.append((node1, node2))
        
        print("=== ARMY-ONLY EDGES (first 10) ===")
        for e in army_only[:10]:
            print("   ", e)
        print(f"Total Army-only edges: {len(army_only)}")
        
        print("\n=== FLEET-ONLY EDGES (first 10) ===")
        for e in fleet_only[:10]:
            print("   ", e)
        print(f"Total Fleet-only edges: {len(fleet_only)}")
        
        print("\n=== ARMY+FLEET EDGES (first 10) ===")
        for e in army_and_fleet[:10]:
            print("   ", e)
        print(f"Total Army+Fleet edges: {len(army_and_fleet)}")

        # 2) Sample adjacency checks picked dynamically from the current map
        both_pair = None
        army_only_pair = None
        fleet_only_pair = None
        coast_water_pair = None

        # Scan edges to find representative samples
        for n1, n2, units in self.connectivity.edges():
            units_set = set(units)
            t1 = self.game.map.area_type(n1)
            t2 = self.game.map.area_type(n2)
            if both_pair is None and units_set == {'A', 'F'}:
                both_pair = (n1, n2)
            if army_only_pair is None and units_set == {'A'} and t1 != 'WATER' and t2 != 'WATER':
                army_only_pair = (n1, n2)
            if fleet_only_pair is None and units_set == {'F'}:
                fleet_only_pair = (n1, n2)
            if coast_water_pair is None:
                coast_cond = ((t1 == 'COAST' and t2 == 'WATER') or (t1 == 'WATER' and t2 == 'COAST')) and 'F' in units_set
                if coast_cond:
                    coast_water_pair = (n1, n2)
            if both_pair and army_only_pair and fleet_only_pair and coast_water_pair:
                break

        known_pairs = [p for p in [both_pair, army_only_pair, fleet_only_pair, coast_water_pair] if p]
        # Fallback if nothing found
        if not known_pairs:
            # Pick up to 3 arbitrary edges
            known_pairs = [(n1, n2) for (n1, n2, _) in self.connectivity.edges()[:3]]

        print("\n=== KNOWN PAIRS UNIT ALLOWANCE ===")
        for t1, t2 in known_pairs:
            if t1 in self.connectivity.graph and t2 in self.connectivity.graph[t1]:
                allowed_units = self.connectivity.get_allowed_units(t1, t2)
                print(f"  {t1} -> {t2}: {allowed_units}")
            else:
                print(f"  {t1} -> {t2}: No direct edge found")

        # 3) Example: for a single territory, show all its neighbors split by unit type
        # Pick an existing territory to inspect: prefer a supply center, else first loc
        territory_to_check = None
        scs = [sc for sc in self.game.map.scs if sc in self.connectivity.graph]
        if scs:
            territory_to_check = scs[0]
        else:
            all_nodes = list(self.connectivity.nodes())
            territory_to_check = all_nodes[0] if all_nodes else None
        if territory_to_check in self.connectivity.graph:
            print(f"\n=== NEIGHBORS OF {territory_to_check} ===")
            for neighbor in sorted(self.connectivity.get_adjacent(territory_to_check)):
                allowed = self.connectivity.get_allowed_units(territory_to_check, neighbor)
                print(f"   {territory_to_check} -> {neighbor}, units = {allowed}")
        else:
            print(f"\n{territory_to_check} is not in the graph?!")
        
    def get_current_phase(self):
        return self.game.get_current_phase()

    def is_game_done(self):
        return self.game.is_game_done

    def step(self):
        """Process the current phase and return what type of phase we end up in."""
        current_phase = self.game.get_current_phase()
        logger.info(f"Processing phase: {current_phase}")

        # Capture centers before
        centers_before = {}
        for pwr_name in self.game.powers:
            code_name = ENGINE_TO_CODE[pwr_name]
            centers = self.game.get_centers(pwr_name)
            if centers:
                centers_before[code_name] = set(centers)

        # Process phase
        if self.game.is_game_done:
            self.done = True
            return None

        # Capture the orders being processed
        processed_orders = {
            ENGINE_TO_CODE[pwr]: orders
            for pwr, orders in self.game.get_orders().items()
        }

        self.game.process()

        # Get detailed results
        order_results = {}
        for pwr_name in self.game.powers:
            code_name = ENGINE_TO_CODE[pwr_name]
            results = self.game.get_order_status(power_name=pwr_name)
            if results:
                order_results[code_name] = results

        # Capture centers after
        centers_after = {}
        for pwr_name in self.game.powers:
            code_name = ENGINE_TO_CODE[pwr_name]
            centers = self.game.get_centers(pwr_name)
            if centers:
                centers_after[code_name] = set(centers)

        # Calculate territory changes
        territory_changes = {}
        for power in set(centers_before.keys()) | set(centers_after.keys()):
            gained = centers_after.get(power, set()) - centers_before.get(power, set())
            lost = centers_before.get(power, set()) - centers_after.get(power, set())
            if gained or lost:
                territory_changes[power] = {
                    'gained': list(gained),
                    'lost': list(lost)
                }

        # Store phase outcomes
        phase_outcomes = {
            'phase': current_phase,
            'orders': processed_orders,
            'results': order_results,
            'territory_changes': territory_changes,
            'dislodged_units': {
                ENGINE_TO_CODE[pwr]: units
                for pwr, units in self.game.get_state().get("dislodged", {}).items()
            }
        }

        if not hasattr(self, "phase_outcomes"):
            self.phase_outcomes = []
        self.phase_outcomes.append(phase_outcomes)
        self.phase_outcomes = self.phase_outcomes[-5:]  # keep at most 5

        new_phase = self.game.get_current_phase()
        return new_phase

    def get_power_names(self):
        """
        Returns the list of power codes (AUT, ENG, FRA, etc.) so that
        the rest of our code can use those as identifiers.
        """
        engine_names = list(self.game.powers.keys())  # e.g. ['AUSTRIA','FRANCE',...]
        return [ENGINE_TO_CODE[n] for n in engine_names]

    def get_game_state_dict(self):
        """
        Return the raw engine state (full names).
        We store that in the JSON file for reloading, so it’s consistent with
        the python-diplomacy library. We do *not* convert to codes here, because
        we want to be able to restore the engine exactly.
        """
        return self.game.get_state()

    def get_recent_moves_summary(self, num_turns=5):
        """
        Returns a compact summary of the last n phases with their orders.
        """
        history = []
        phases = self.game.get_phase_history()

        recent_phases = phases[-num_turns:] if len(phases) >= num_turns else phases

        for phase_data in recent_phases:
            phase_name = phase_data.name
            if not phase_data.orders:
                print('!! No orders recorded')
                continue

            moves = {}
            for eng_power, orders in phase_data.orders.items():
                code_power = ENGINE_TO_CODE.get(eng_power, eng_power)
                moves[code_power] = orders

            if moves:
                history.append({
                    'phase': phase_name,
                    'moves': moves
                })
        return history

    def get_adjacency_map(self):
        """Returns a complete adjacency map with territory types."""
        adjacency_map = {}
        for loc in self.game.map.locs:
            adjacency_map[loc] = {
                'type': self.game.map.loc_type[loc],
                'adjacent': self.game.map.abut_list(loc),
                'is_supply_center': loc in self.game.map.scs
            }
        return adjacency_map

    def validate_moves(self, power_name, moves_dict):
        """
        Validates a dictionary of moves against the game engine.
        Returns only the moves that the engine accepts.
        All moves are converted to uppercase.
        
        Args:
            power_name: The power code (e.g. 'FRA')
            moves_dict: Dictionary of unit -> list of possible moves
                    e.g. {'A PAR': ['A PAR - BUR', 'A PAR S A MAR']}
        
        Returns:
            Dict of the same format but only containing valid moves
        """
        import time
        start_time = time.time()
        
        engine_power = CODE_TO_ENGINE[power_name]
        
        # Convert everything to uppercase
        upper_moves_dict = {unit.upper(): [move.upper() for move in moves] 
                        for unit, moves in moves_dict.items()}
        
        # Flatten all moves and track which unit they belong to
        all_moves = []
        move_to_unit = {}
        for unit, unit_moves in upper_moves_dict.items():
            for move in unit_moves:
                all_moves.append(move)
                move_to_unit[move] = unit
        
        # Test each order individually to get normalized form
        valid_moves = {}
        valid_count = 0
        
        for move in all_moves:
            self.game.clear_orders(engine_power)
            self.game.set_orders(engine_power, [move])
            stored = self.game.get_orders(engine_power)
            normalized = stored[0] if stored else None
            
            if normalized:
                valid_count += 1
                unit = move_to_unit[move]
                if unit not in valid_moves:
                    valid_moves[unit] = []
                valid_moves[unit].append(normalized)
            #else:
                #print(move)
        
        validation_time = time.time() - start_time
        invalid_count = len(all_moves) - valid_count
        #logger.info(f"Move validation took {validation_time:.2f} seconds for {len(all_moves)} moves. {invalid_count} out of {len(all_moves)} moves were invalid ({(invalid_count/len(all_moves)*100):.1f}%)")
        
        self.game.clear_orders(engine_power)

        return valid_moves

    def get_valid_moves(self, power_name, include_holds=False):
        """Returns all valid moves for a given power in the current phase.
        
        - Removes duplicate moves before returning.
        - Optionally includes hold (`H`) moves if `include_holds=True`.
        """

        phase = self.game.get_current_phase()
        phase_type = phase[-1] if phase else "?"
        engine_power = self.game.powers[CODE_TO_ENGINE[power_name]]

        if not engine_power or self.game.is_game_done:
            return {}

        occupied_territories = {}  # Stores which power occupies which territory
        for p in self.game.powers.values():
            for unit in p.units:
                unit_loc = unit.split(' ')[1].split('/')[0]  # Remove coast indicators
                occupied_territories[unit_loc] = unit  # Store occupying unit

        valid_moves = {}

        if phase_type == 'M':
            # ==================
            # Movement phase
            # ==================
            for unit in engine_power.units:
                unit_type, unit_loc = unit.split(' ', 1)
                possible_moves = set()  # Use set to remove duplicates automatically

                for loc in self.game.map.locs:
                    move_indicator = ""

                    # Check if this move would enter an occupied space
                    if loc in occupied_territories:
                        move_indicator = " (Occupied!)"

                    # Basic moves
                    if self.game.map.abuts(unit_type, unit_loc, '-', loc):
                        possible_moves.add(f'{unit_type} {unit_loc}-{loc}{move_indicator}')

                    # Support holds
                    if self.game.map.abuts(unit_type, unit_loc, 'S', loc):
                        if loc in occupied_territories:  # Only add if there's a unit there
                            possible_moves.add(f'{unit_type} {unit_loc} S {loc}')
                        
                        # Support moves
                        for dest in self.game.map.abut_list(loc):
                            if self.game.map.abuts(unit_type, unit_loc, 'S', dest):
                                if loc in occupied_territories:  # Only add if there's a unit that could move
                                    possible_moves.add(f'{unit_type} {unit_loc} S {loc}-{dest}')

                # Add hold move if `include_holds=True`
                if include_holds:
                    possible_moves.add(f'{unit_type} {unit_loc} H')

                # Store all valid moves for this unit (converted back to sorted list)
                valid_moves[unit] = sorted(possible_moves)
            valid_moves = self.validate_moves(power_name, valid_moves)
            return valid_moves

        elif phase_type == 'R':
            # ==================
            # Retreat phase
            # ==================
            for unit, retreat_locs in engine_power.retreats.items():
                possible_moves = set()  # Use set to remove duplicates

                for loc in retreat_locs:
                    move_indicator = ""
                    if loc in occupied_territories:
                        move_indicator = " (Occupied!)"

                    possible_moves.add(f"{unit} R {loc}{move_indicator}")

                possible_moves.add(f"{unit} D")  # Disband option
                valid_moves[unit] = sorted(possible_moves)
            valid_moves = self.validate_moves(power_name, valid_moves)
            print('valid retreat moves for', power_name)
            print(valid_moves)
            return valid_moves

        elif phase_type == 'A':
            # ==================
            # Adjustment phase
            # ==================
            valid_moves = {}
            num_centers = len(engine_power.centers)
            num_units = len(engine_power.units)
            build_difference = num_centers - num_units

            if build_difference > 0:
                build_moves = set()
                for home_sc in engine_power.homes:
                    if home_sc not in engine_power.centers:  # Must currently own it
                        continue
                    if home_sc in occupied_territories:  # Must be unoccupied
                        continue

                    is_occupied = home_sc in occupied_territories
                    move_indicator = " (Occupied!)" if is_occupied else ""

                    if self.game.map.area_type(home_sc) in ('COAST', 'BICOAST', 'SEA'):
                        build_moves.add(f"B A {home_sc}{move_indicator}")
                        build_moves.add(f"B F {home_sc}{move_indicator}")
                    else:
                        build_moves.add(f"B A {home_sc}{move_indicator}")

                valid_moves["BUILDS"] = sorted(build_moves)

            elif build_difference < 0:
                disband_moves = set()
                for unit in engine_power.units:
                    unit_type, unit_loc = unit.split()  # Splits e.g. "A PAR" into "A" and "PAR"
                    disband_moves.add(f"R {unit}")  # or could use REMOVE instead of R
                valid_moves["DISBANDS"] = sorted(disband_moves)
            print(valid_moves)
            #valid_moves = self.validate_moves(power_name, valid_moves)
            print(valid_moves)
            return valid_moves

        else:
            return {}


    def get_strategic_overview(self, power_code):
        """
        Returns a dictionary keyed by territory, containing the old functionality (status, adjacency,
        occupant, valid moves for each unit) plus BFS expansions for:
        - nearest_friendly_unit
        - nearest_unowned_sc
        - combined BFS if shorter for armies
        Also includes an 'adjustments' dict if it's the adjustment phase, listing possible builds/disbands.
        
        Newly added:
        - For each adjacent territory, a `can_support` list of all units (including other powers) that
        can issue a support ("S") order to that territory.
        - A `nearest` entry for each territory, with:
        - `units_not_ours`: up to 3 nearest non-friendly units (distance and path).
        - `held_scs_not_ours`: up to 3 nearest supply centers owned by other powers (distance and path).
        """

        current_phase = self.game.get_current_phase()  # e.g. "S1901M"
        phase_type = current_phase[-1] if current_phase else "?"

        engine_power = self.game.powers[CODE_TO_ENGINE[power_code]]
        our_centers = set(engine_power.centers)
        our_units_set = {u.split(' ')[1].split('/')[0].upper() for u in engine_power.units}
        all_scs = set(self.game.map.scs)

        # territory -> engine-owner
        territory_owner = {}
        for pwr_name, pwr_obj in self.game.powers.items():
            for c in pwr_obj.centers:
                territory_owner[c.upper()] = pwr_name

        unowned_scs = {sc.upper() for sc in all_scs if sc.upper() not in territory_owner}

        # --- Gather dislodged or build sets if relevant ---
        dislodged_terr_set = set()
        if phase_type == 'R':
            retreats_for_power = self.game.get_state().get('retreats', {}).get(CODE_TO_ENGINE[power_code], [])
            for d_unit in retreats_for_power:
                d_terr = d_unit.split(' ',1)[1].split('/')[0].upper()
                dislodged_terr_set.add(d_terr)

        build_terr_set = set()
        if phase_type == 'A':
            adjustments_for_pwr = self.game.get_state().get('adjustments', {}).get(CODE_TO_ENGINE[power_code], None)
            if adjustments_for_pwr:
                if isinstance(adjustments_for_pwr, dict):
                    adjustments_for_pwr = [adjustments_for_pwr]
                for adj_task in adjustments_for_pwr:
                    build_locs = adj_task.get('locations', [])
                    disband_units = adj_task.get('disband', [])
                    for bl in build_locs:
                        build_terr_set.add(bl.upper())
                    for du in disband_units:
                        terr_part = du.split(' ',1)[1].split('/')[0].upper()
                        build_terr_set.add(terr_part)

        # Combine all territories we want to show
        territories_to_analyze = (
            set(t.upper() for t in our_centers)
            .union(our_units_set)
            .union(dislodged_terr_set)
            .union(build_terr_set)
        )

        

        overview = {}

        # ========== 1) Build territory-level info (status, occupant, adjacency) + BFS expansions ==========
        for terr in territories_to_analyze:
            base_terr = terr.upper()

            # territory type
            terr_type = None
            if base_terr in self.game.map.loc_type:
                terr_type = self.game.map.loc_type[base_terr]
            elif base_terr.lower() in self.game.map.loc_type:
                terr_type = self.game.map.loc_type[base_terr.lower()]

            # occupant units (any power)
            occupying_units = []
            for p_obj in self.game.powers.values():
                for unit_str in p_obj.units:
                    u_loc_base = unit_str.split(' ', 1)[1].split('/')[0].upper()
                    if u_loc_base == base_terr:
                        occupying_units.append(unit_str)

            # controlling power if SC
            controlling_power_code = None
            if base_terr in territory_owner:
                controlling_power_code = ENGINE_TO_CODE[territory_owner[base_terr]]

            # adjacency
            raw_adjs_up = self.game.map.abut_list(base_terr, incl_no_coast=False)
            raw_adjs_lo = self.game.map.abut_list(base_terr.lower(), incl_no_coast=False)
            raw_adjs = set(raw_adjs_up + raw_adjs_lo)

            adj_list = []
            for adj_loc in raw_adjs:
                adj_base = adj_loc.split('/')[0].upper()
                adj_type = None
                if adj_base in self.game.map.loc_type:
                    adj_type = self.game.map.loc_type[adj_base]
                elif adj_base.lower() in self.game.map.loc_type:
                    adj_type = self.game.map.loc_type[adj_base.lower()]

                adj_occ_units = []
                for pwr in self.game.powers.values():
                    for u_str in pwr.units:
                        uu_loc = u_str.split(' ',1)[1].split('/')[0].upper()
                        if uu_loc == adj_base:
                            adj_occ_units.append(u_str)

                adj_cp = None
                if adj_base in territory_owner:
                    adj_cp = ENGINE_TO_CODE[territory_owner[adj_base]]

                adj_list.append({
                    "name": adj_base,
                    "type": adj_type,
                    "is_supply_center": (adj_base in all_scs),
                    "controlling_power": adj_cp,
                    "occupying_units": adj_occ_units
                })

            # status: contested if occupant differs from controlling power
            status = None
            if base_terr in all_scs and controlling_power_code:
                occupant_codes = set()
                for u_str in occupying_units:
                    for p_name, pwr_obj in self.game.powers.items():
                        if u_str in pwr_obj.units:
                            occupant_codes.add(ENGINE_TO_CODE[p_name])
                if occupant_codes and occupant_codes != {controlling_power_code}:
                    status = (
                        f"CONTESTED - Supply center controlled by {controlling_power_code}, "
                        f"occupied by {', '.join(occupant_codes)}"
                    )

            overview[base_terr] = {
                "type": terr_type,
                "is_supply_center": (base_terr in all_scs),
                "controlling_power": controlling_power_code,
                "status": status,
                "occupying_units": occupying_units,
                "adjacent_territories": sorted(adj_list, key=lambda x: x["name"]),
                "units": {}
            }

        # ========== 2) BFS expansions for each of our units (nearest_friendly_center, etc.) ==========
        all_friendly_occupant_map = {}
        for u_str in engine_power.units:
            u_loc = u_str.split(' ',1)[1].split('/')[0].upper()
            all_friendly_occupant_map[u_loc] = u_str

        for unit_str in engine_power.units:
            unit_type, loc_part = unit_str.split(' ',1)
            base_loc = loc_part.split('/')[0].upper()

            if base_loc not in overview:
                overview[base_loc] = {
                    "type": None,
                    "is_supply_center": (base_loc in all_scs),
                    "controlling_power": None,
                    "status": None,
                    "occupying_units": [unit_str],
                    "adjacent_territories": [],
                    "units": {}
                }

            def match_friendly_sc(t):
                return t if t in (x.upper() for x in our_centers) else None

            def match_unowned_sc(t):
                return t if t in unowned_scs else None

            occupant_map = {k:v for (k,v) in all_friendly_occupant_map.items() if v != unit_str}
            normal_types = {'F'} if unit_type == 'F' else {'A'}

            # BFS to nearest friendly SC
            path_fc, match_fc = self.bfs_shortest_path(base_loc, match_friendly_sc, normal_types)
            # BFS to unowned SC
            path_usc, match_usc = self.bfs_shortest_path(base_loc, match_unowned_sc, normal_types)
            # BFS to adjacency of friendly occupant
            path_fu, (fu_terr, fu_unit) = self.bfs_nearest_adjacent(base_loc, occupant_map, normal_types)

            on_unowned = (base_loc in unowned_scs)
            # Combined BFS if Army
            combined_types = {'A','F'} if unit_type == 'A' else normal_types

            cb_path_fc, cb_match_fc = self.bfs_shortest_path(base_loc, match_friendly_sc, combined_types)
            fc_combined = None
            if cb_path_fc and (not path_fc or len(cb_path_fc)<len(path_fc)):
                fc_combined = {"distance": len(cb_path_fc)-1, "path": cb_path_fc}

            cb_path_usc, cb_match_usc = self.bfs_shortest_path(base_loc, match_unowned_sc, combined_types)
            usc_combined = None
            if cb_path_usc and (not path_usc or len(cb_path_usc)<len(path_usc)):
                usc_combined = {"distance": len(cb_path_usc)-1, "path": cb_path_usc}

            cb_path_fu, (cb_fu_terr, cb_fu_unit) = self.bfs_nearest_adjacent(base_loc, occupant_map, combined_types)
            fu_combined = None
            if cb_path_fu and (not path_fu or len(cb_path_fu)<len(path_fu)):
                fu_combined = {
                    "distance": len(cb_path_fu)-1,
                    "path": cb_path_fu,
                    "matched_territory": cb_fu_terr,
                    "matched_unit": cb_fu_unit
                }

            overview[base_loc]["units"][unit_str] = {
                "nearest_friendly_center": {
                    "distance": (len(path_fc)-1 if path_fc else None),
                    "path": path_fc,
                    "matched_territory": match_fc,
                    "by_convoy": fc_combined
                },
                "nearest_friendly_unit": {
                    "distance": (len(path_fu)-1 if path_fu else None),
                    "path": path_fu,
                    "matched_territory": fu_terr,
                    "matched_unit": fu_unit,
                    "by_convoy": fu_combined
                },
                "nearest_unowned_sc": {
                    "distance": (len(path_usc)-1 if path_usc else None),
                    "path": path_usc,
                    "matched_territory": match_usc,
                    "by_convoy": usc_combined
                },
                "on_unowned_sc": on_unowned,
                # We'll add valid_moves below
            }

        # ========== 3) Attach "valid_moves" (movement/retreat) to each of our units ==========
        valid_moves_dict = self.get_valid_moves(power_code, include_holds=False)
        for terr_key in overview:
            for unit_key in overview[terr_key]["units"]:
                if unit_key in valid_moves_dict:
                    overview[terr_key]["units"][unit_key]["valid_moves"] = valid_moves_dict[unit_key]
                else:
                    overview[terr_key]["units"][unit_key]["valid_moves"] = []

       
        # ========== [NEW] BFS for up to 3 nearest occupant units not ours, and up to 3 nearest SC not ours ==========
        def _find_up_to_three_nearest_occupants_not_ours(start, unit_type=None):
            """
            Find up to 3 nearest units not belonging to us.
            
            Args:
                start: Starting territory
                unit_type: If provided, use only paths valid for this unit type ('A' or 'F')
            """
            results = []
            visited = set([start])
            queue = deque([(start, [start], 0)])  # (territory, path, distance)
            
            # Determine allowed unit types for movement
            if unit_type == 'A':
                allowed_unit_types = {'A'}
            elif unit_type == 'F':
                allowed_unit_types = {'F'}
            else:
                allowed_unit_types = {'A', 'F'}
            
            while queue and len(results) < 3:
                current, path, dist = queue.popleft()
                
                # Check occupant units here
                for pwr_name, pwr_obj in self.game.powers.items():
                    if ENGINE_TO_CODE[pwr_name] == power_code:
                        continue  # skip our own
                    for occupant_unit in pwr_obj.units:
                        unit_loc = occupant_unit.split(' ', 1)[1].split('/')[0].upper()
                        if unit_loc == current:
                            results.append({
                                "distance": dist,
                                "path": path,
                                "unit": f"{occupant_unit} ({ENGINE_TO_CODE[pwr_name]})"
                            })
                            if len(results) >= 3:
                                break
                    if len(results) >= 3:
                        break
                
                if len(results) >= 3:
                    break
                    
                # expand BFS - only using valid edges for our unit type
                if current in self.connectivity.graph:
                    for neighbor in self.connectivity.graph[current]:
                        edge_unit_types = self.connectivity.get_allowed_units(current, neighbor)
                        if edge_unit_types.intersection(allowed_unit_types) and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor], dist + 1))
            
            return results

        def _find_up_to_three_nearest_sc_held_by_others(start, unit_type=None):
            """
            Find up to 3 nearest supply centers not controlled by us (includes unowned SCs).
            
            Args:
                start: Starting territory
                unit_type: If provided, use only paths valid for this unit type ('A' or 'F')
            """
            results = []
            visited = set([start])
            queue = deque([(start, [start], 0)])  # (territory, path, distance)
            
            # Determine allowed unit types for movement
            if unit_type == 'A':
                allowed_unit_types = {'A'}
            elif unit_type == 'F':
                allowed_unit_types = {'F'}
            else:
                allowed_unit_types = {'A', 'F'}
            
            while queue and len(results) < 3:
                current, path, dist = queue.popleft()
                
                # Check if it's a supply center that we don't control
                if current in all_scs:
                    controlling_power = None
                    if current in territory_owner:
                        controlling_power = ENGINE_TO_CODE[territory_owner[current]]
                    
                    # Include if either unowned or owned by someone else
                    if controlling_power != power_code:
                        results.append({
                            "distance": dist,
                            "path": path,
                            "territory": current,
                            "controlled_by": controlling_power or "Unowned"
                        })
                        if len(results) >= 3:
                            break
                            
                # expand BFS - only using valid edges for our unit type
                if current in self.connectivity.graph:
                    for neighbor in self.connectivity.graph[current]:
                        edge_unit_types = self.connectivity.get_allowed_units(current, neighbor)
                        if edge_unit_types.intersection(allowed_unit_types) and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor], dist + 1))
            
            return results

        # Attach "nearest" summary to each territory
        for base_terr in overview.keys():
            if base_terr == "adjustments":
                continue
                
            # Determine unit type if one of our units is here
            unit_type = None
            for unit in overview[base_terr].get('occupying_units', []):
                if unit in engine_power.units:
                    unit_type = unit.split()[0]  # 'A' or 'F'
                    break
                    
            # Call BFS with appropriate unit type
            nearest_units_not_ours = _find_up_to_three_nearest_occupants_not_ours(base_terr, unit_type)
            nearest_scs_not_ours = _find_up_to_three_nearest_sc_held_by_others(base_terr, unit_type)
            
            overview[base_terr]["nearest"] = {
                "units_not_ours": nearest_units_not_ours,
                "held_scs_not_ours": nearest_scs_not_ours
            }

        # ========== [NEW] Add 'can_support' info to each adjacent territory ==========
        for territory in overview:
            adj_list = overview[territory]["adjacent_territories"]
            for adjacency_info in adj_list:
                adj_base = adjacency_info["name"]
                can_support_units = []
                # Check all units on the board that can "S" the territory 'adj_base'
                for pwr_name, pwr_obj in self.game.powers.items():
                    for occupant_unit in pwr_obj.units:
                        occupant_type, occupant_loc = occupant_unit.split(' ',1)
                        occupant_loc = occupant_loc.split('/')[0].upper()
                        if self.game.map.abuts(occupant_type, occupant_loc, 'S', adj_base):
                            can_support_units.append(f"{occupant_unit} ({ENGINE_TO_CODE[pwr_name]})")
                adjacency_info["can_support"] = can_support_units

         # ========== 4) If adjustment phase, store "BUILDS" or "DISBANDS" in overview["adjustments"] ==========
        overview["adjustments"] = {}
        if phase_type == 'A':
            if "BUILDS" in valid_moves_dict:
                overview["adjustments"]["BUILDS"] = valid_moves_dict["BUILDS"]
            if "DISBANDS" in valid_moves_dict:
                overview["adjustments"]["DISBANDS"] = valid_moves_dict["DISBANDS"]
            if "NO_ADJUSTMENT" in valid_moves_dict:
                overview["adjustments"]["NO_ADJUSTMENT"] = valid_moves_dict["NO_ADJUSTMENT"]


        return overview


    def get_strategic_overview_text(self, power_code):
        # 1) Gather all territory info + BFS expansions from get_strategic_overview
        overview = self.get_strategic_overview(power_code)
        engine_power = self.game.powers[CODE_TO_ENGINE[power_code]]

        header = f"Strategic Overview for {power_code}"
        current_phase = self.game.get_current_phase()
        if current_phase and current_phase[0] == 'W':
            # Possibly show build/remove info in the header
            num_centers = len(engine_power.centers)
            num_units = len(engine_power.units)
            diff = num_centers - num_units
            if diff > 0:
                header += f"\n{power_code} can build {diff} unit{'s' if diff>1 else ''}"
            elif diff < 0:
                header += f"\n{power_code} must remove {abs(diff)} unit{'s' if abs(diff)>1 else ''}"

        header += """


            Territories where you have units or supply centers, plus shortest-path expansions and valid moves.
            """
        lines = [header]

        # 2) Sort territories: supply centers first, then alphabetical
        sorted_territories = sorted(
            [t for t in overview.keys() if t != "adjustments" and '/' not in t],
            key=lambda x: (not overview[x]['is_supply_center'], x)
        )

        # 3) For each territory, print core info (status, occupant, adjacency) + BFS expansions + valid moves
        for territory in sorted_territories:
            data = overview[territory]
            lines.append(f"\n# Strategic territory held by {power_code}: {territory} ({data['type'] or 'Unknown'})")

            if data['status']:
                lines.append(f"Status: {data['status']}")

            # If it's your home center in Movement phase and you occupy it, warn
            if data['is_supply_center'] and territory in engine_power.homes:
                if current_phase and current_phase[-1] == 'M':
                    occupant_codes = []
                    for occ_unit in data['occupying_units']:
                        for p_name, p_obj in self.game.powers.items():
                            if occ_unit in p_obj.units:
                                occupant_codes.append(ENGINE_TO_CODE[p_name])
                    occupant_codes = set(occupant_codes)
                    if power_code in occupant_codes:
                        lines.append("Status: WARNING: Your unit in this home SC will block building here in Winter.")

            if data['is_supply_center']:
                ctrl = data['controlling_power'] or 'None'
                lines.append(f"Supply Center - Controlled by: {ctrl}")

            if data['occupying_units']:
                occupant_strs = []
                for u_str in data['occupying_units']:
                    occupant_power_code = None
                    for p_name, p_obj in self.game.powers.items():
                        if u_str in p_obj.units:
                            occupant_power_code = ENGINE_TO_CODE[p_name]
                            break
                    occupant_strs.append(f"{u_str} ({occupant_power_code})")
                lines.append("Units present: " + ", ".join(occupant_strs))

            if 'units' in data and data['units']:
                for unit_name, bfs_info in data['units'].items():
                    if unit_name in engine_power.units:
                        lines.append(f"  Shortest path for {unit_name}:")
                        if bfs_info.get('on_unowned_sc'):
                            lines.append("    => Currently on an unowned supply center!")

                        # nearest_friendly_unit
                        fu = bfs_info['nearest_friendly_unit']
                        if fu['distance'] is not None:
                            min_distance = fu['distance']
                            matched_territory = fu['matched_territory']
                            primary_unit = fu['matched_unit'] 
                            
                            # Display primary unit and any other units at the same location
                            friendly_units_at_location = []
                            if primary_unit:
                                friendly_units_at_location.append(primary_unit)
                            
                            # Check for other units at this location
                            for other_unit in engine_power.units:
                                if other_unit == unit_name or other_unit == primary_unit:
                                    continue
                                    
                                unit_loc = other_unit.split(' ', 1)[1].split('/')[0].upper()
                                if unit_loc == matched_territory:
                                    friendly_units_at_location.append(other_unit)
                            
                            # Display all units at this location
                            if friendly_units_at_location:
                                path_display = fu['path'][:] if fu['path'] else []
                                if matched_territory not in path_display:
                                    path_display.append(matched_territory)
                                    
                                lines.append("    => Nearest friendly unit:")
                                for friendly_unit in friendly_units_at_location:
                                    lines.append(f"       {friendly_unit} ({power_code}) path={path_display}")

                        # Show "valid_moves"
                        if 'valid_moves' in bfs_info and bfs_info['valid_moves']:
                            lines.append("    => Possible moves:")
                            for mv in bfs_info['valid_moves']:
                                # Check if this is a movement order and extract destination
                                parts = mv.split()
                                occupancy_info = ""
                                
                                if '-' in mv and len(parts) >= 3:
                                    # For movement orders: extract destination
                                    dest_index = parts.index('-') + 1 if '-' in parts else -1
                                    if dest_index > 0 and dest_index < len(parts):
                                        dest = parts[dest_index].split('/')[0].upper()
                                        # Check if destination is occupied
                                        for pwr_name, pwr_obj in self.game.powers.items():
                                            for occ_unit in pwr_obj.units:
                                                occ_loc = occ_unit.split(' ')[1].split('/')[0].upper()
                                                if occ_loc == dest:
                                                    occ_power = ENGINE_TO_CODE[pwr_name]
                                                    occupancy_info = f" (Occupied by {occ_power})"
                                                    break
                                            if occupancy_info:
                                                break
                                
                                lines.append(f"       {mv}{occupancy_info}")

            # Check if this territory has one of our units on it
            our_unit_here = None
            for unit in data.get('occupying_units', []):
                if unit in engine_power.units:
                    our_unit_here = unit
                    break

            # Only show nearest units if we have a unit here or it's our SC without units
            show_nearest_units = our_unit_here or (data['is_supply_center'] and data['controlling_power'] == power_code)
            
            if show_nearest_units and "nearest" in data:
                nr_info = data["nearest"]
                units_not_ours = nr_info.get("units_not_ours", [])
                
                if units_not_ours:
                    lines.append("  Nearest units (not ours):")
                    for item in units_not_ours:
                        # Format: * F SEV (RUS), dist=2, path=[SMY→ARM→SEV]
                        unit = item["unit"]
                        dist = item["distance"]
                        path = item["path"]
                        
                        # Make path display more compact with arrows
                        path_display = "→".join(path)
                        lines.append(f"    {unit}, path=[{path_display}]")
                
                # Only show nearest SCs if we have a unit here (skip if it's just our SC)
                if our_unit_here and "held_scs_not_ours" in nr_info:
                    scs_not_ours = nr_info.get("held_scs_not_ours", [])
                    if scs_not_ours:
                        lines.append("  Nearest supply centers (not controlled by us):")
                        for item in scs_not_ours:
                            # Format: * SEV (RUS), dist=2, path=[SMY→ARM→SEV]
                            terr = item["territory"]
                            owner = item["controlled_by"]
                            dist = item["distance"]
                            path = item["path"]
                            
                            # Make path display more compact with arrows
                            path_display = "→".join(path)
                            lines.append(f"    {terr} ({owner}), dist={dist}, path=[{path_display}]")

            lines.append("Adjacent territories (including units that can support/move to the adjacent territory):")
            for adjt in data['adjacent_territories']:
                adj_line = [adjt["name"], f"({adjt['type'] or 'UNKNOWN'})"]
                if adjt['is_supply_center']:
                    adj_line.append(f"SC Control: {adjt['controlling_power'] or 'None'}")
                if adjt['occupying_units']:
                    occ_strs = []
                    for ou in adjt['occupying_units']:
                        oc = None
                        for pn, p_obj in self.game.powers.items():
                            if ou in p_obj.units:
                                oc = ENGINE_TO_CODE[pn]
                                break
                        occ_strs.append(f"{ou} ({oc})")
                    adj_line.append("Units: " + ", ".join(occ_strs))

                lines.append("  " + " ".join(adj_line))
                # Filter out units from current territory in can_support list
                if "can_support" in adjt and adjt["can_support"]:
                    # Get base territory name (remove /SC, /NC, etc.)
                    base_territory = territory.split('/')[0].upper()
                    
                    filtered_support = []
                    for supporter in adjt["can_support"]:
                        # Get unit territory and strip coastal indicators
                        unit_parts = supporter.split(' ')
                        if len(unit_parts) >= 2:
                            unit_location = unit_parts[1].split('/')[0].upper()
                            # Only include if not from current territory
                            if unit_location != base_territory:
                                filtered_support.append(supporter)
                        else:
                            # If we can't parse it, include it
                            filtered_support.append(supporter)
                            
                    if filtered_support:
                        lines.append(f"    => Can support/move to: {', '.join(filtered_support)}")

            # 4) If it's the adjustment phase, show "BUILDS" or "DISBANDS" from overview["adjustments"]
            if current_phase and len(current_phase) > 1 and current_phase[-1] == 'A':
                if "adjustments" in overview:
                    adj_block = overview["adjustments"]
                    if adj_block:
                        lines.append("\nAdjustment Phase Orders:")
                        if "BUILDS" in adj_block and adj_block["BUILDS"]:
                            lines.append("  Possible Builds:")
                            for build_order in adj_block["BUILDS"]:
                                lines.append(f"    {build_order}")
                        if "DISBANDS" in adj_block and adj_block["DISBANDS"]:
                            lines.append("  Possible Disbands:")
                            for d_order in adj_block["DISBANDS"]:
                                lines.append(f"    {d_order}")
                        if "NO_ADJUSTMENT" in adj_block and adj_block["NO_ADJUSTMENT"]:
                            lines.append("  No adjustment needed. Moves:")
                            for nmv in adj_block["NO_ADJUSTMENT"]:
                                lines.append(f"    {nmv}")

        overview_text = "\n".join(lines)
        #print(overview_text)
        return overview_text




    def get_last_phase_outcomes_text(self):
        """
        Returns a text summary of the outcomes for ALL phases in the current year.
        
        Example: if the last stored phase is F1901R, this gathers S1901M, S1901R,
        S1901A, F1901M, F1901R, etc. (all phases that share '1901').
        """
        if not hasattr(self, "phase_outcomes") or not self.phase_outcomes:
            return ""

        # Identify the year of the *latest* recorded phase
        last_record = self.phase_outcomes[-1]
        last_phase_name = last_record['phase']  # e.g. "S1901M", "F1901R", etc.

        if len(last_phase_name) < 6:
            # fallback: just summarize the final record if we can't parse properly
            return self._summarize_single_phase_outcome(last_record)

        # Extract the year substring (e.g. '1901' from 'S1901M')
        year_str = last_phase_name[1:5]

        # Gather all stored outcomes that match this year
        matching_year_outcomes = [
            ph for ph in self.phase_outcomes
            if len(ph['phase']) >= 6 and ph['phase'][1:5] == year_str
        ]

        # Sort them in chronological order (S=0, F=1, W=2; M=0, R=1, A=2)
        def phase_sort_key(ph):
            p = ph['phase']
            season_order = {'S': 0, 'F': 1, 'W': 2}
            subphase_order = {'M': 0, 'R': 1, 'A': 2}
            season_val = season_order.get(p[0], 99)
            subphase_val = subphase_order.get(p[-1], 99)
            year_val = 9999
            if p[1:5].isdigit():
                year_val = int(p[1:5])
            return (year_val, season_val, subphase_val)

        matching_year_outcomes.sort(key=phase_sort_key)

        # Build a textual summary for all matching phases
        text = f"=== Outcomes for {year_str} ===\n\n"
        for phase_data in matching_year_outcomes:
            phase_name = phase_data['phase']
            text += f"--- Phase {phase_name} ---\n\n"
            text += self._summarize_single_phase_outcome(phase_data)
            text += "\n"

        #print(text)
        return text


    def _summarize_single_phase_outcome(self, phase_data):
        """
        Helper to summarize a single phase outcome in text form.
        Shows each order and the final result.
        Also includes territory changes and dislodged units at the end.
        """
        text = "Orders & Results:\n"
        # phase_data should have keys: 'orders', 'results', etc.

        # 1) For each power, show the submitted orders:
        for power_name in sorted(phase_data['orders'].keys()):
            orders_for_power = phase_data['orders'][power_name] or []
            results_for_power = phase_data['results'].get(power_name, {})
            text += f"{power_name}:\n"
            #text += f"Orders Provided: {orders_for_power}\n"

            for order_str in orders_for_power:
                # Example order_str: "A SER - BUL", "F ENG C A LON-BRE", etc.
                # We'll take the first two tokens as the unit name, e.g. "A SER"
                tokens = order_str.split()
                if len(tokens) >= 2:
                    # The first two tokens are something like ["A","SER"] or ["F","GRE"]
                    unit_name = tokens[0] + " " + tokens[1]
                else:
                    unit_name = order_str  # fallback

                # Get the result string for that unit from results_for_power
                outcome = results_for_power.get(unit_name, "")
                if not outcome:
                    outcome_text = "successful"
                else:
                    outcome_text = str(outcome)

                text += f"  {order_str} => {outcome_text}\n"

        # 2) Territory Changes
        territory_changes = phase_data.get('territory_changes', {})
        if territory_changes:
            text += "\nTerritory Changes:\n"
            for power_name, changes in territory_changes.items():
                gained = changes.get('gained', [])
                lost = changes.get('lost', [])
                if gained:
                    text += f"  {power_name} gained: {', '.join(gained)}\n"
                if lost:
                    text += f"  {power_name} lost: {', '.join(lost)}\n"

        # 3) Dislodged Units
        dislodged = phase_data.get('dislodged_units', {})
        if any(dislodged.values()):
            text += "\nDislodged Units:\n"
            for pwr, units in dislodged.items():
                if units:
                    text += f"  {pwr}: {', '.join(units)}\n"
        
        return text


    def get_observation_for_power(self, power_name):
        """
        Return the observable state for an agent in code form.
        power_name is expected to be a code, e.g. 'ENG'.
        """
        engine_state = self.game.get_state()
        code_state = _engine_state_to_code_state(engine_state)

        strategic_text = self.get_strategic_overview_text(power_name)
        outcomes_text = self.get_last_phase_outcomes_text()

        obs = {
            "phase": self.game.get_current_phase(),
            "board_state": code_state,
            "your_power": power_name,
            "power_codes": self.get_power_names(),
            "recent_moves": self.get_recent_moves_summary(2),
            "adjacency_map": self.get_adjacency_map(),
            "valid_moves": self.get_valid_moves(power_name),
            "strategic_overview": strategic_text,
            "last_turn_outcomes": outcomes_text
        }
        return obs

    def set_orders(self, power_name, orders):
        """
        Set orders for a power and return validation results with normalization mappings.
        
        Args:
            power_name: Power code (e.g. 'FRA')
            orders: List of orders to set
            
        Returns:
            tuple[dict, list]:
                - order_mappings: Dict mapping raw orders to their normalized forms (or None if invalid)
                - accepted_orders: List of orders that were actually accepted by the engine
        """
        engine_pwr = CODE_TO_ENGINE.get(power_name)
        if not engine_pwr:
            logger.warning(f"Unknown code power {power_name} cannot set orders.")
            return {}, []

        if not orders:
            logger.warning(f"No orders set for {power_name}!")
            return {}, []

        # Test each order individually to get its normalized form
        order_mappings = {}
        for raw_order in orders:
            self.game.clear_orders(engine_pwr)
            self.game.set_orders(engine_pwr, [raw_order])
            stored = self.game.get_orders(engine_pwr)
            order_mappings[raw_order] = stored[0] if stored else None
        
        # Now set all valid orders
        valid_orders = [norm for norm in order_mappings.values() if norm is not None]
        logger.info(f"Setting validated orders for {power_name}: {valid_orders}")
        self.game.clear_orders(engine_pwr)
        self.game.set_orders(engine_pwr, orders)
        
        return order_mappings, valid_orders

    def compute_score(self):
        """
        Compute a simple "score" as the number of supply centers, but return
        a dict keyed by codes for external consumption/logging.
        """
        supply_centers = self.game.get_state()['centers']
        scores = {}
        for eng_pwr, centers in supply_centers.items():
            code_pwr = ENGINE_TO_CODE[eng_pwr]
            scores[code_pwr] = len(centers)
        return scores

    def to_engine(self, power_code: str) -> str:
        """Convert a 3-letter (or generated) power code to engine power name."""
        return CODE_TO_ENGINE[power_code]

    def to_code(self, engine_power_name: str) -> str:
        """Convert an engine power name to our code."""
        return ENGINE_TO_CODE[engine_power_name]
