import argparse
import os
import re
import sys
from urllib.request import urlopen


def fetch_text(source: str) -> str:
    if source.startswith('http://') or source.startswith('https://'):
        with urlopen(source) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    with open(source, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def normalize_coast_suffix(name: str) -> str:
    s = name.lower()
    if 'north coast' in s:
        return 'NC'
    if 'south coast' in s:
        return 'SC'
    if 'east coast' in s:
        return 'EC'
    if 'west coast' in s:
        return 'WC'
    return ''


def make_abbrev_map(names):
    """Generate unique 3-letter abbreviations; for coasts append /NC,/SC,/EC,/WC."""
    base_codes = {}
    used = set()

    def base_code(full_name: str) -> str:
        letters = ''.join(c for c in full_name.upper() if c.isalpha())
        code = (letters[:3] if len(letters) >= 3 else (letters + 'XXX')[:3])
        i = 2
        dedup = code
        while dedup in used:
            dedup = (code[:2] + str(i))[:3]
            i += 1
        used.add(dedup)
        return dedup

    # First pass: assign bases
    for name in names:
        # Strip coast decorations for base
        base = name
        if ' (' in name and name.endswith(')'):
            base = name.split(' (', 1)[0].strip()
        if base not in base_codes:
            base_codes[base] = base_code(base)

    # Second pass: assemble final mapping
    mapping = {}
    for name in names:
        if ' (' in name and name.endswith(')'):
            base = name.split(' (', 1)[0].strip()
            suffix = normalize_coast_suffix(name)
            mapping[name] = f"{base_codes[base]}/{suffix}" if suffix else base_codes[base]
        else:
            mapping[name] = base_codes[name]
    return mapping


def parse_territories(text: str):
    # Match array('Name','Type','Yes|No', countryId, x, y, sx, sy)
    terr_re = re.compile(r"array\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*(\d+)\s*,\s*[^)]*\)")
    terrs = []
    for m in terr_re.finditer(text):
        name, ttype, supply, country_id = m.group(1), m.group(2), m.group(3), int(m.group(4))
        terrs.append({
            'name': name,
            'type': ttype,  # Sea|Coast|Land
            'supply': (supply.strip().lower() == 'yes'),
            'country': country_id,
        })
    return terrs


def parse_borders(text: str):
    # Match array('From','To','Yes|No','Yes|No')
    bord_re = re.compile(r"array\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")
    borders = []
    for m in bord_re.finditer(text):
        frm, to, fleets, armies = m.group(1), m.group(2), m.group(3), m.group(4)
        borders.append({
            'from': frm,
            'to': to,
            'fleets': fleets.strip().lower() == 'yes',
            'armies': armies.strip().lower() == 'yes',
        })
    return borders


def type_to_map(ttype: str) -> str:
    up = ttype.strip().lower()
    if up == 'sea':
        return 'WATER'
    if up == 'coast':
        return 'COAST'
    if up == 'land':
        return 'LAND'
    return 'LAND'


def to_titlecase(abbrev: str) -> str:
    # Titlecase per map spec (first uppercase, rest lowercase)
    return abbrev[:1].upper() + abbrev[1:].lower()


def build_map_lines(territories, borders, map_name: str, victory: int, country_names=None):
    # Collect names
    all_names = [t['name'] for t in territories]
    name_to_abbr = make_abbrev_map(all_names)

    # Index terrains and SCs
    name_to_type = {t['name']: type_to_map(t['type']) for t in territories}
    supply_names = {t['name'] for t in territories if t['supply']}
    country_ids = sorted({t['country'] for t in territories if t['country'] is not None})

    # Multi-coast bases
    base_to_coasts = {}
    for n in all_names:
        if ' (' in n and n.endswith(')'):
            base = n.split(' (', 1)[0].strip()
            base_to_coasts.setdefault(base, []).append(n)

    # Build aggregated adjacency flags: from_token -> to_token -> (fleets, armies)
    adj_flags = {}
    def agg_adj(frm, to, fleets, armies):
        if not fleets and not armies:
            return
        d = adj_flags.setdefault(frm, {})
        f, a = d.get(to, (False, False))
        d[to] = (f or fleets, a or armies)

    # Helper to map a name to a single token: coast -> coast token; base -> lowercase token
    def map_name_to_token(name: str) -> str:
        if ' (' in name and name.endswith(')'):
            return name_to_abbr[name]
        base = name
        if base in base_to_coasts:
            return name_to_abbr[base].lower()
        return name_to_abbr[base]

    # Add borders both ways for undirected movement
    for b in borders:
        f_name, t_name = b['from'], b['to']
        if f_name not in name_to_abbr or t_name not in name_to_abbr:
            continue
        f_abbr = map_name_to_token(f_name)
        t_abbr = map_name_to_token(t_name)
        agg_adj(f_abbr, t_abbr, b['fleets'], b['armies'])
        agg_adj(t_abbr, f_abbr, b['fleets'], b['armies'])

    # Enforce symmetry for all adjacencies except when from-token is lowercase base and to-token is WATER
    tokens = list(adj_flags.keys())
    for a in list(adj_flags.keys()):
        for b, (f,a_flag) in list(adj_flags[a].items()):
            # If reverse missing and not allowed one-way, add it
            reverse = adj_flags.get(b, {}).get(a)
            a_is_lower_base = (a == a.lower()) and (a.upper()[:3] == a[:3].upper())
            b_is_water = b in {name_to_abbr[n] for n in all_names if name_to_type[n] == 'WATER'}
            if reverse is None and not (a_is_lower_base and b_is_water):
                agg_adj(b, a, f, a_flag)

    lines = []
    # Do not emit MAP directive to avoid self-include errors when loading from a custom path
    lines.append(f"VICTORY {victory}")

    # Aliases
    for name in sorted(all_names):
        ab = name_to_abbr[name]
        alias = name.replace(' ', '+')
        lines.append(f"{name.upper()} = {ab} {alias}")

    # Coast entries must come before main base for multi-coasts
    def emit_loc_line(name: str):
        ab = name_to_abbr[name]
        ttype = name_to_type[name]
        keyword = 'COAST' if ttype == 'COAST' else ('WATER' if ttype == 'WATER' else 'LAND')
        lines.append(f"{keyword} {ab}")

    # Emit coasts first
    # Precompute set of water abbrs
    water_abbrs = {name_to_abbr[n] for n in all_names if name_to_type[n] == 'WATER'}

    for base, coasts in sorted(base_to_coasts.items()):
        # Coast tokens (e.g., TUS/EC) first
        for c in sorted(coasts):
            emit_loc_line(c)
        # Then the base as lowercase token to indicate multi-coast province
        base_ab = name_to_abbr[base]
        low_token = base_ab.lower()
        ttype = name_to_type.get(base, 'COAST')
        # Per spec, the lowercase base must be adjacent to all WATER neighbors of its coasts
        water_neighbors = set()
        for c in coasts:
            cab = name_to_abbr[c]
            for to_tok, (f,a) in adj_flags.get(cab, {}).items():
                if to_tok in water_abbrs:
                    water_neighbors.add(to_tok)
        # Emit adjacency flags for base to those waters (both unit types ok for adjacency symmetry)
        for w in water_neighbors:
            # Only base (lowercase) -> WATER is needed; reverse is intentionally omitted
            agg_adj(low_token, w, True, True)
        # Build ABUTS text for base from the aggregated flags
        base_abuts = []
        for to_tok, (f,a) in sorted(adj_flags.get(low_token, {}).items()):
            if f and a:
                base_abuts.append(to_tok)
            elif a and not f:
                base_abuts.append(to_tok.lower())  # armies only
            elif f and not a:
                base_abuts.append(to_titlecase(to_tok))  # fleets only
        abuts = ' ABUTS ' + ' '.join(base_abuts) if base_abuts else ' '
        keyword = 'COAST' if ttype == 'COAST' else ('WATER' if ttype == 'WATER' else 'LAND')
        lines.append(f"{keyword} {low_token}{abuts}")

    # Emit non-multicoast locations
    for name in sorted(all_names):
        if name in base_to_coasts:
            continue
        if ' (' in name and name.endswith(')'):
            continue
        emit_loc_line(name)

    # Now emit ABUTS for coast and non-multicoast tokens using aggregated flags
    # Replace earlier simple emit_loc_line neighbors with aggregated flags
    rebuilt = []
    for i, line in enumerate(lines):
        if line.startswith('COAST ') or line.startswith('LAND ') or line.startswith('WATER '):
            parts = line.split()
            if not parts:
                continue
            token = parts[1]
            # Recompute ABUTS
            abuts_list = []
            for to_tok, (f,a) in sorted(adj_flags.get(token, {}).items()):
                if f and a:
                    abuts_list.append(to_tok)
                elif a and not f:
                    abuts_list.append(to_tok.lower())  # armies only
                elif f and not a:
                    abuts_list.append(to_titlecase(to_tok))  # fleets only
            prefix = ' '.join(parts[:2])
            new_line = prefix + (' ABUTS ' + ' '.join(abuts_list) if abuts_list else '')
            rebuilt.append(new_line)
        else:
            rebuilt.append(line)
    lines = rebuilt

    # Political: powers and starts
    # 0 is typically neutral/unassigned in vDip installer data
    sc_abbrs = {name_to_abbr[n] for n in supply_names}
    neutral_scs = sorted([name_to_abbr[n] for n in supply_names if any(t['name'] == n and t['country'] == 0 for t in territories)])
    if neutral_scs:
        lines.append("UNOWNED " + ' '.join(neutral_scs))

    # Country names from variant.php if provided
    id_to_name = {}
    if country_names:
        for idx, nm in enumerate(country_names, start=1):
            id_to_name[idx] = nm.split()[0].upper()
    else:
        for cid in sorted({t['country'] for t in territories if t['country'] > 0}):
            id_to_name[cid] = f"POWER_{cid}"

    # Homes by power
    power_homes = {id_to_name[cid]: [] for cid in id_to_name}
    for t in territories:
        if t['supply'] and t['country'] in id_to_name:
            power_homes[id_to_name[t['country']]].append(name_to_abbr[t['name']])

    # Prepare initial units per power from homes: Fleet on coastal homes (with water); Army on land
    base_to_water_coast = {}
    for base, coasts in base_to_coasts.items():
        candidates = []
        for c in coasts:
            cab = name_to_abbr[c]
            for to_tok, (f,a) in adj_flags.get(cab, {}).items():
                if to_tok in water_abbrs and f:
                    candidates.append(cab)
                    break
        base_to_water_coast[base] = candidates

    # Reverse abbr -> name for base lookup
    abbr_to_name = {v: k for k, v in name_to_abbr.items()}

    power_units = {p: [] for p in power_homes}
    for pwr in power_homes:
        placed_any_army = False
        for home in sorted(power_homes[pwr]):
            base_name = abbr_to_name.get(home, None)
            if not base_name:
                continue
            ttype = name_to_type.get(base_name, 'LAND')
            if ttype == 'LAND':
                power_units[pwr].append(f"A {home}")
                placed_any_army = True
            else:
                # COAST -> fleet if possible
                coast_choices = base_to_water_coast.get(base_name, [])
                if coast_choices:
                    power_units[pwr].append(f"F {coast_choices[0]}")
                else:
                    power_units[pwr].append(f"A {home}")
                    placed_any_army = True
        if not placed_any_army and power_homes[pwr]:
            h = sorted(power_homes[pwr])[0]
            # Ensure at least one army exists
            power_units[pwr].append(f"A {h}")

    # Emit powers with homes (assign valid, unique single-letter abbreviations not 'M' or '?')
    used_abbrevs = set()
    def choose_abbrev(pname: str) -> str:
        for ch in pname:
            if ch.isalpha():
                up = ch.upper()
                if up not in ('M', '?') and up not in used_abbrevs:
                    used_abbrevs.add(up)
                    return up
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if c not in ('M', '?') and c not in used_abbrevs:
                used_abbrevs.add(c)
                return c
        return 'X'

    for pwr in power_homes:
        homes = sorted(power_homes[pwr])
        if not homes:
            continue
        # Ownership word: first word of power name (no spaces), abbreviation: its first letter
        ownword = pwr.split()[0]
        abbrev = choose_abbrev(ownword)
        lines.append(f"{pwr} ({ownword}:{abbrev}) " + ' '.join(homes))
        # Emit UNITS block for initial units under current power context
        units_list = power_units.get(pwr, [])
        if units_list:
            lines.append("UNITS")
            for unit in units_list:
                lines.append(unit)


    return lines


def main():
    parser = argparse.ArgumentParser(description="Convert vDiplomacy install-save PHP into python-diplomacy .map")
    parser.add_argument('--source', required=False, help='Local file or URL to install-save PHP (default: --url)')
    parser.add_argument('--url', required=False, help='URL to vDiplomacy install-save PHP')
    parser.add_argument('--out', required=True, help='Output path for .map')
    parser.add_argument('--map-name', default='treaty_of_verdun', help='Map name identifier')
    parser.add_argument('--victory', type=int, default=10, help='Victory center target')
    parser.add_argument('--variant-url', default='https://www.vdiplomacy.com/dev/files_helper.php?&variantID=58&action=view&file=variant.php&basedir=/', help='URL to variant.php for country names')
    args = parser.parse_args()

    if not args.source and not args.url:
        print('Provide either --source or --url')
        sys.exit(1)

    src = args.source or args.url
    text = fetch_text(src)

    territories = parse_territories(text)
    borders = parse_borders(text)

    if not territories or not borders:
        print('Failed to parse territories or borders from source.')
        sys.exit(2)

    # Fetch country names
    try:
        variant_txt = fetch_text(args.variant_url)
        m = re.search(r"countries\s*=\s*array\(([^)]*)\)\s*;", variant_txt)
        country_names = []
        if m:
            inner = m.group(1)
            country_names = re.findall(r"'([^']+)'", inner)
    except Exception:
        country_names = []

    lines = build_map_lines(territories, borders, args.map_name, args.victory, country_names=[n for n in country_names])

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote map to {out_path}')


if __name__ == '__main__':
    main()


