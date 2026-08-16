import pretty_midi
import random

def create_strum_midi_3sec():
    pm = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=25)  # Acoustic Guitar (steel)

    # C 和弦音符對照 (Pitch, String Number)
    bass_note = (48, 5)            # t: 5 弦 3 品 (C3 / MIDI 48)
    treble_notes = [(55, 3), (60, 2), (64, 1)] # p3, p2, p1

    def add_stroke(instrument, notes, start_time, direction="down", strum_ms=22, base_vel=85, note_duration=0.6):
        """
        strum_ms: 刷弦的微小時間差 (毫秒)
        """
        num_notes = len(notes)
        if num_notes == 0:
            return

        reverse_sort = True if direction == "down" else False
        sorted_notes = sorted(notes, key=lambda x: x[1], reverse=reverse_sort)

        delay_sec = (strum_ms / 1000.0) / max(num_notes - 1, 1)

        for i, (pitch, string_num) in enumerate(sorted_notes):
            n_start = start_time + (i * delay_sec)
            n_end = n_start + note_duration
            
            vel = max(30, min(127, base_vel + random.randint(-3, 3)))
            
            note = pretty_midi.Note(
                velocity=vel,
                pitch=pitch,
                start=n_start,
                end=n_end
            )
            instrument.notes.append(note)

    # ==========================================
    # 節奏時間軸配置 (總長 5.0 秒 / 80 BPM)
    # ==========================================
    # 1 拍 = 0.75 秒，半拍 = 0.375 秒
    #
    # 0.000s : t    (第 1 拍：根音)
    # 0.750s : d321 (第 2 拍：正拍下刷)
    # 1.125s : u321 (第 2 拍半：反拍上刷)
    # 1.500s : d321 (第 3 拍：正拍下刷)
    # 1.875s : u321 (第 3 拍半：反拍上刷)
    # 2.250s : d321 (第 4 拍：收尾下刷，餘音延伸)

    # 1. 0.000s: t (根音 C3，延長持續發音)
    guitar.notes.append(pretty_midi.Note(
        velocity=95,
        pitch=bass_note[0],
        start=0.000,
        end=2.000
    ))

    # 2. 0.750s: d321 (第 2 拍下刷)
    add_stroke(guitar, treble_notes, start_time=0.750, direction="down", strum_ms=70, base_vel=86, note_duration=0.7)

    # 3. 1.125s: u321 (第 2 拍半上刷，力道稍輕)
    add_stroke(guitar, treble_notes, start_time=1.125, direction="up", strum_ms=60, base_vel=72, note_duration=0.5)

    # 4. 1.500s: d321 (第 3 拍下刷)
    add_stroke(guitar, treble_notes, start_time=1.500, direction="down", strum_ms=70, base_vel=84, note_duration=0.7)

    # 5. 1.875s: u321 (第 3 拍半上刷)
    add_stroke(guitar, treble_notes, start_time=1.875, direction="up", strum_ms=60, base_vel=70, note_duration=0.5)

    # 6. 2.250s: d321 (第 4 拍收尾下刷)
    add_stroke(guitar, treble_notes, start_time=2.250, direction="down", strum_ms=90, base_vel=90, note_duration=0.75)

    # ==========================================
    # 慢速演示：下刷 (d321) 與上刷 (u321) 各 1 秒展延
    # 方便確認三弦發音順序是否正確
    # ==========================================
    # 下刷 (d321): 低音弦→高音弦，耗時 1.0s
    add_stroke(guitar, treble_notes, start_time=3.0, direction="down", strum_ms=1000, base_vel=85, note_duration=0.8)

    # 上刷 (u321): 高音弦→低音弦，耗時 1.0s
    add_stroke(guitar, treble_notes, start_time=5.0, direction="up", strum_ms=1000, base_vel=85, note_duration=0.8)

    pm.instruments.append(guitar)
    pm.write("strum_3sec_80bpm.mid")

if __name__ == "__main__":
    create_strum_midi_3sec()