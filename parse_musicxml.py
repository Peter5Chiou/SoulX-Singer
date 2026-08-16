import json
import xml.etree.ElementTree as ET
from typing import Any

NS = {"m": "http://www.musesoftware.com/ns/musicxml"}  # unused but harmless


def parse_musicxml(file_path: str) -> dict:
    tree = ET.parse(file_path)
    root = tree.getroot()

    divisions = 1
    time_beats = None
    time_beat_type = None
    bpm = None

    measures_data: list[dict] = []
    current_beat = 1.0  # running beat position across measures

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            measure_num = int(measure.get("number", 0))

            chords: List[dict] = []
            melody: List[dict] = []

            measure_start_beat = current_beat

            for elem in measure:
                tag = elem.tag

                if tag == "attributes":
                    div_el = elem.find("divisions")
                    if div_el is not None and div_el.text:
                        divisions = int(div_el.text)

                    time_el = elem.find("time")
                    if time_el is not None:
                        beats_el = time_el.find("beats")
                        btype_el = time_el.find("beat-type")
                        if beats_el is not None and beats_el.text:
                            time_beats = int(beats_el.text)
                        if btype_el is not None and btype_el.text:
                            time_beat_type = int(btype_el.text)

                elif tag == "direction":
                    sound_el = elem.find("sound")
                    if sound_el is not None:
                        tempo_attr = sound_el.get("tempo")
                        if tempo_attr is not None:
                            try:
                                bpm = float(tempo_attr)
                            except ValueError:
                                pass

                elif tag == "harmony":
                    root_el = elem.find("root")
                    kind_el = elem.find("kind")
                    if root_el is not None:
                        step_el = root_el.find("root-step")
                        alter_el = root_el.find("root-alter")
                        root_text = step_el.text if step_el is not None else ""

                        kind_text = kind_el.get("text") if kind_el is not None else ""
                        kind_val = kind_el.text if kind_el is not None else ""

                        if kind_text:
                            chord_symbol = root_text + kind_text
                        elif kind_val == "minor":
                            chord_symbol = root_text + "m"
                        elif kind_val in ("major", "augmented", "diminished"):
                            chord_symbol = root_text
                        elif kind_val == "dominant":
                            chord_symbol = root_text + "7"
                        elif kind_val:
                            chord_symbol = f"{root_text}{kind_val}"
                        else:
                            chord_symbol = root_text

                        chords.append({
                            "symbol": chord_symbol,
                            "beat": round(current_beat, 3),
                            "duration_beats": 0.0,
                        })

                elif tag == "note":
                    is_rest = elem.find("rest") is not None
                    if is_rest:
                        dur_el = elem.find("duration")
                        if dur_el is not None and dur_el.text:
                            dur = int(dur_el.text)
                            current_beat += dur / divisions
                        continue

                    pitch_el = elem.find("pitch")
                    dura_el = elem.find("duration")
                    if pitch_el is not None and dura_el is not None and dura_el.text:
                        step_el = pitch_el.find("step")
                        octave_el = pitch_el.find("octave")
                        if step_el is not None and step_el.text and octave_el is not None and octave_el.text:
                            pitch_str = f"{step_el.text}{octave_el.text}"
                            dur = int(dura_el.text)
                            duration_beats = dur / divisions
                            melody.append({
                                "pitch": pitch_str,
                                "beat": round(current_beat, 3),
                                "duration_beats": round(duration_beats, 3),
                            })
                            current_beat += duration_beats
                        else:
                            # Pitch incomplete, skip but still advance beat
                            dur = int(dura_el.text)
                            current_beat += dur / divisions
                    elif dura_el is not None and dura_el.text:
                        dur = int(dura_el.text)
                        current_beat += dur / divisions

            for i, ch in enumerate(chords):
                next_beat = chords[i + 1]["beat"] if i + 1 < len(chords) else current_beat
                ch["duration_beats"] = round(next_beat - ch["beat"], 3)

            measures_data.append({
                "measure_number": measure_num,
                "chords": chords,
                "melody": melody,
            })

            measure_start_beat = current_beat

    time_sig = f"{time_beats}/{time_beat_type}" if time_beats and time_beat_type else ""

    result = {
        "time_signature": time_sig,
        "measures": measures_data,
    }
    if bpm is not None:
        result["bpm"] = bpm

    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    path = sys.argv[1] if len(sys.argv) > 1 else "微聲盼望-Viola.musicxml"
    data = parse_musicxml(path)
    out_path = Path(path).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")