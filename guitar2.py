import pretty_midi
import random

def create_fast_strum_midi():
    pm = pretty_midi.PrettyMIDI()
    # 使用 General MIDI Program 25: Acoustic Guitar (steel)
    guitar = pretty_midi.Instrument(program=25)

    # C 和弦音符對照 (Pitch, String Number)
    # t = 5 弦 3 品 (C3 / MIDI 48)
    # p3 = 3 弦 0 品 (G3 / MIDI 55), p2 = 2 弦 1 品 (C4 / MIDI 60), p1 = 1 弦 0 品 (E4 / MIDI 64)
    bass_note = (48, 5)
    treble_notes = [(55, 3), (60, 2), (64, 1)]

    def add_stroke(instrument, notes, start_time, direction="down", strum_ms=18, base_vel=85, note_duration=0.3):
        """
        strum_ms: 3 條弦總共掃過的極短微延遲（下刷 18ms，上刷 12ms）
        """
        num_notes = len(notes)
        if num_notes == 0:
            return

        # 下刷 (down): 3弦 -> 2弦 -> 1弦
        # 上刷 (up):   1弦 -> 2弦 -> 3弦
        reverse_sort = True if direction == "down" else False
        sorted_notes = sorted(notes, key=lambda x: x[1], reverse=reverse_sort)

        # 單弦微幅時間差 (毫秒轉秒)
        delay_sec = (strum_ms / 1000.0) / max(num_notes - 1, 1)

        for i, (pitch, string_num) in enumerate(sorted_notes):
            n_start = start_time + (i * delay_sec)
            n_end = n_start + note_duration
            
            # 隨機人性化微調 (±3 MIDI Velocity)
            vel = max(30, min(127, base_vel + random.randint(-3, 3)))
            
            note = pretty_midi.Note(
                velocity=vel,
                pitch=pitch,
                start=n_start,
                end=n_end
            )
            instrument.notes.append(note)

    # ==========================================
    # 節奏時間軸配置 (總長 1.2 秒)
    # ==========================================
    # 0.00s : t    (第 1 拍：根音)
    # 0.30s : d321 (第 2 拍：下刷)
    # 0.45s : u321 (第 2 拍半：上刷)
    # 0.60s : d321 (第 3 拍：下刷)
    # 0.75s : u321 (第 3 拍半：上刷)
    # 0.90s : d321 (第 4 拍：重音下刷收尾延伸至 1.2s)

    # 1. 0.00s: t (根音 C3，扎實的音量)
    guitar.notes.append(pretty_midi.Note(
        velocity=96,
        pitch=bass_note[0],
        start=0.00,
        end=0.80
    ))

    # 2. 0.30s: d321 (強拍下刷，耗時 18ms)
    add_stroke(guitar, treble_notes, start_time=0.30, direction="down", strum_ms=18, base_vel=88)

    # 3. 0.45s: u321 (反拍上刷，較輕較快，耗時 12ms)
    add_stroke(guitar, treble_notes, start_time=0.45, direction="up", strum_ms=12, base_vel=72)

    # 4. 0.60s: d321 (正拍下刷，耗時 18ms)
    add_stroke(guitar, treble_notes, start_time=0.60, direction="down", strum_ms=18, base_vel=85)

    # 5. 0.75s: u321 (反拍上刷，較輕，耗時 12ms)
    add_stroke(guitar, treble_notes, start_time=0.75, direction="up", strum_ms=12, base_vel=70)

    # 6. 0.90s: d321 (第 4 拍重音下刷收尾，自然響到 1.30s 結束)
    add_stroke(guitar, treble_notes, start_time=0.90, direction="down", strum_ms=22, base_vel=92, note_duration=0.4)

    pm.instruments.append(guitar)
    pm.write("fast_strum_1.2s.mid")

if __name__ == "__main__":
    create_fast_strum_midi()