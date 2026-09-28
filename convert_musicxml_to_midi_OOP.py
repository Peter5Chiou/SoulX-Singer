#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MusicXML to MIDI & Repeat Analysis Converter (OOP Architecture)"""

from __future__ import annotations
import math
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import music21
from mido import MidiFile, MidiTrack, MetaMessage, Message
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ==============================================================================
# 1. 領域模型 (Domain Models / Data Transfer Objects)
# ==============================================================================

@dataclass
class LyricProcessor:
    """負責處理歌詞文字清洗與對齊之公用工具"""

    @staticmethod
    def clean(text: Optional[str]) -> str:
        """只保留文字（字母/漢字），移除數字、標點符號與空白。"""
        return ''.join(ch for ch in (text or '') if ch.isalpha())

    @classmethod
    def map_by_verse(cls, lyrics: List[music21.note.Lyric]) -> Dict[int, str]:
        """將 music21 的 Lyric 物件清單轉為 {段落編號: 歌詞}，避免延伸音節覆蓋真實文字。"""
        mapping: Dict[int, str] = {}
        for lyr in lyrics:
            num = lyr.number
            txt = cls.clean(lyr.text)
            if num not in mapping or (txt and not mapping[num]):
                mapping[num] = txt
        return mapping


@dataclass
class NoteEvent:
    """樂譜中最原始的音符/休止符事件（尚未展開反覆）"""
    pitch: Optional[int]                 # MIDI 音高 (0-127)，若為 None 代表休止符
    quarter_length: float                # 時值（四分音符為單位）
    lyrics: Dict[int, str] = field(default_factory=dict)  # {verse_number: text}
    measure_number: Optional[int] = None

    @property
    def is_rest(self) -> bool:
        return self.pitch is None


@dataclass
class ExpandedNote:
    """按照演奏順序展開後的單一音符事件"""
    pitch: Optional[int]
    quarter_length: float
    lyric_text: str
    measure_number: Optional[int]
    verse_number: int
    is_refrain: bool

    @property
    def is_rest(self) -> bool:
        return self.pitch is None


@dataclass
class RepeatMarker:
    """反覆記號或跳躍標籤資訊"""
    measure: int
    direction: str
    end_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'measure': self.measure,
            'direction': self.direction,
            'end_s': round(self.end_s, 3)
        }


@dataclass
class NavigationInfo:
    """樂曲導航結構 (如 D.S. al Fine / D.C. al Fine)"""
    nav_type: str
    segno_m: int
    fine_m: int
    jump_m: int


@dataclass
class Segment:
    """JSON 輸出的分段資訊"""
    seg: int
    start_s: float
    end_s: float
    start_ms: int
    end_ms: int
    lyrics_verse: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'seg': self.seg,
            'start_s': round(self.start_s, 3),
            'end_s': round(self.end_s, 3),
            'start_ms': self.start_ms,
            'end_ms': self.end_ms,
            'lyrics_verse': self.lyrics_verse,
        }


@dataclass
class ScoreMetadata:
    """樂譜整體資訊（速度、拍號與弱起拍）"""
    bpm: int = 120
    time_signature_num: int = 4
    time_signature_den: int = 4
    pickup_beats: float = 0.0  # 弱起小節拍數 (quarterLength)，非弱起樂曲為 0.0
    last_bar_beats: float = 0.0  # 最後一小節實際拍數，若為滿小節則為 0.0
    
    @property
    def second_per_beat(self) -> float:
        return 60.0 / self.bpm


# ==============================================================================
# 2. 樂譜解析層 (Parser Layer)
# ==============================================================================

class MusicXmlParser:
    """負責透過 music21 解析 MusicXML，提取拍號、速度與連續音符（合併 Tie）"""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self._score = music21.converter.parse(xml_path)

    def get_parts(self) -> Dict[str, music21.stream.Part]:
        """取得所有聲部，格式為 {聲部名稱: Part 物件}"""
        parts_dict: Dict[str, music21.stream.Part] = {}
        for p in self._score.parts:
            name = p.partName if p.partName else f"Track - {p.id}"
            parts_dict[name] = p
        return parts_dict

    def extract_metadata(self, part: music21.stream.Part) -> ScoreMetadata:
          """解析指定聲部的 BPM、主拍號、弱起拍與結尾小節時值"""
          bpm = 120
          try:
            tempo_marks = (
                self._score.recurse().getElementsByClass(
                    music21.tempo.MetronomeMark
                )
            )
            if tempo_marks:
              mark = tempo_marks[0]
              if hasattr(mark, 'getQuarterBPM'):
                bpm = int(round(mark.getQuarterBPM()))
              elif hasattr(mark, '_number') and mark._number is not None:
                bpm = int(mark._number)
              elif hasattr(mark, 'bpm'):
                bpm = int(mark.bpm)
          except Exception as e:
            print(f'[Warn] BPM 解析失敗，使用預設值 120: {e}')

          num, den = 4, 4
          try:
            ts_list = list(
                part.recurse().getElementsByClass(music21.meter.TimeSignature)
            )
            if ts_list:
              num = ts_list[0].numerator
              den = ts_list[0].denominator
          except Exception as e:
            print(f'[Warn] 拍號讀取失敗，使用預設值 4/4: {e}')

          pickup_beats = 0.0
          last_bar_beats = 0.0
          measures = list(part.getElementsByClass(music21.stream.Measure))
          bar_dur = float(num * (4.0 / den))  # 滿小節拍數（quarterLength）

          if measures:
            # 1. 檢測第一小節（弱起）
            first_m = measures[0]
            first_dur = sum(
                float(el.duration.quarterLength)
                for el in first_m.notesAndRests
                if el.duration.quarterLength > 0
            )
            if 0 < first_dur < bar_dur:
              pickup_beats = first_dur

            # 2. 檢測最後一小節（結尾不完整小節）
            last_m = measures[-1]
            last_dur = sum(
                float(el.duration.quarterLength)
                for el in last_m.notesAndRests
                if el.duration.quarterLength > 0
            )
            if 0 < last_dur < bar_dur:
              last_bar_beats = last_dur  # 例如 3/4 拍只有 2 拍，last_bar_beats = 2.0

          return ScoreMetadata(
              bpm=bpm,
              time_signature_num=num,
              time_signature_den=den,
              pickup_beats=pickup_beats,
              last_bar_beats=last_bar_beats,
          )

    def parse_note_events(self, part: music21.stream.Part) -> List[NoteEvent]:
        """提取連續音符與休止符，保留 MuseScore 式弱起（直接從真實音符開始），並合併延音 (Tie)"""
        notes_list: List[NoteEvent] = []

        all_elements = part.flatten().notesAndRests
        valid_elements = [
            e for e in all_elements
            if isinstance(e, (music21.note.Note, music21.note.Rest)) and e.duration.quarterLength > 0
        ]
        valid_elements.sort(key=lambda e: e.offset)

        if not valid_elements:
            return []

        cur_pitch: Optional[int] = None
        cur_dur: float = 0.0
        cur_lyrics_raw: List[music21.note.Lyric] = []
        cur_m_num = getattr(valid_elements[0], 'measureNumber', None)

        def flush():
            nonlocal cur_pitch, cur_dur, cur_lyrics_raw, cur_m_num
            if cur_dur > 0:
                notes_list.append(
                    NoteEvent(
                        pitch=cur_pitch,
                        quarter_length=cur_dur,
                        lyrics=LyricProcessor.map_by_verse(cur_lyrics_raw),
                        measure_number=cur_m_num,
                    )
                )
            cur_pitch, cur_dur, cur_lyrics_raw = None, 0.0, []

        for i, el in enumerate(valid_elements):
            m_num = getattr(el, 'measureNumber', None)
            is_tie = False

            if i > 0:
                prev = valid_elements[i - 1]
                if isinstance(el, music21.note.Note) and isinstance(prev, music21.note.Note):
                    same_pitch = (el.pitch == prev.pitch)
                    contiguous = (abs(float(el.offset) - float(prev.offset + prev.duration.quarterLength)) < 1e-4)

                    if same_pitch and contiguous:
                        el_tie = getattr(el.tie, 'type', None)
                        prev_tie = getattr(prev.tie, 'type', None)
                        if el_tie in ('stop', 'continue') or prev_tie in ('start', 'continue'):
                            is_tie = True

            if isinstance(el, music21.note.Rest):
                flush()
                notes_list.append(
                    NoteEvent(
                        pitch=None,
                        quarter_length=float(el.duration.quarterLength),
                        lyrics={},
                        measure_number=m_num,
                    )
                )
            elif is_tie:
                cur_dur += float(el.duration.quarterLength)
                if el.lyrics:
                    cur_lyrics_raw.extend(el.lyrics)
            else:
                flush()
                cur_pitch = el.pitch.midi
                cur_dur = float(el.duration.quarterLength)
                cur_lyrics_raw = list(el.lyrics) if el.lyrics else []
                cur_m_num = m_num

        flush()
        return notes_list


# ==============================================================================
# 3. 樂曲反覆與導航分析層 (Repeat & Navigation Analysis)
# ==============================================================================

class RepeatAnalyzer:
    """分析小節線與記號（反覆記號、Segno、Fine、D.S. al Fine、D.C. al Fine）"""

    def __init__(self, part: music21.stream.Part, second_per_beat: float):
        self.part = part
        self.second_per_beat = second_per_beat

    def analyze(self) -> Tuple[List[RepeatMarker], bool, Optional[NavigationInfo]]:
        markers: List[RepeatMarker] = []
        complex_found = False

        ds_m, dc_m, fine_m, segno_m = None, None, None, 1

        for m in self.part.getElementsByClass(music21.stream.Measure):
            m_num = int(m.number)
            m_offset_s = round(float(m.offset) * self.second_per_beat, 3)

            # 檢測表達式記號 (Expressions)
            for el in m.recurse():
                text_val = str(getattr(el, 'text', '') or getattr(el, 'name', '') or getattr(el, 'content', '')).strip().lower()

                if isinstance(el, music21.repeat.DalSegnoAlFine) or any(k in text_val for k in ('dal segno al fine', 'd.s. al fine', 'd.s.al fine')):
                    ds_m = m_num
                    markers.append(RepeatMarker(m_num, 'd.s. al fine', m_offset_s))
                elif isinstance(el, music21.repeat.DaCapoAlFine) or any(k in text_val for k in ('da capo al fine', 'd.c. al fine', 'd.c.al fine')):
                    dc_m = m_num
                    markers.append(RepeatMarker(m_num, 'd.c. al fine', m_offset_s))
                elif isinstance(el, music21.repeat.Fine) or text_val == 'fine':
                    fine_m = m_num
                    markers.append(RepeatMarker(m_num, 'fine', m_offset_s))
                elif isinstance(el, music21.repeat.Segno) or text_val == 'segno':
                    segno_m = m_num
                    markers.append(RepeatMarker(m_num, 'segno', m_offset_s))
                elif isinstance(el, music21.repeat.RepeatExpression):
                    complex_found = True

            # 檢測反覆小節線 (Barlines)
            for bl in (getattr(m, 'leftBarline', None), getattr(m, 'rightBarline', None)):
                if bl is None:
                    continue
                reps = []
                if isinstance(bl, music21.bar.Repeat):
                    reps.append(bl)
                elif hasattr(bl, 'repeat') and getattr(bl, 'repeat'):
                    rep = getattr(bl, 'repeat')
                    reps.extend(rep if isinstance(rep, (list, tuple)) else [rep])

                for r in reps:
                    d = getattr(r, 'direction', None)
                    d = {'start': 'forward', 'end': 'backward', 'start-end': 'forward-backward'}.get(d, d)
                    dirs = ['forward', 'backward'] if d == 'forward-backward' else ([d] if isinstance(d, str) else [])
                    for direction in dirs:
                        markers.append(RepeatMarker(m_num, direction, m_offset_s))

                if getattr(bl, 'ending', None):
                    complex_found = True

        nav_info = None
        if ds_m is not None and fine_m is not None:
            nav_info = NavigationInfo(nav_type='ds_al_fine', segno_m=segno_m, fine_m=fine_m, jump_m=ds_m)
        elif dc_m is not None and fine_m is not None:
            nav_info = NavigationInfo(nav_type='dc_al_fine', segno_m=1, fine_m=fine_m, jump_m=dc_m)

        return markers, complex_found, nav_info


# ==============================================================================
# 4. 演出順序展開層 (Playback Sequence Expander)
# ==============================================================================

class PlaybackExpander:
    """將音符依主副歌、反覆跳躍標籤展開為線性播放順序"""

    def __init__(self, notes: List[NoteEvent], second_per_beat: float):
        self.notes = notes
        self.second_per_beat = second_per_beat

    def _get_active_verses(self) -> List[int]:
        """計算總共存在且有歌詞的段落號碼 (Verse Numbers)"""
        max_verse = 1
        for n in self.notes:
            for v_num in n.lyrics.keys():
                if isinstance(v_num, int) and v_num > max_verse:
                    max_verse = v_num

        active_verses = [
            v for v in range(1, max_verse + 1)
            if any(n.lyrics.get(v, "") for n in self.notes)
        ]
        return active_verses or [1]

    def expand(self, markers: List[RepeatMarker], complex_found: bool,
               nav_info: Optional[NavigationInfo]) -> Tuple[List[Segment], List[ExpandedNote]]:
        verse_order = self._get_active_verses()
        segments: List[Segment] = []
        expanded_notes: List[ExpandedNote] = []

        if nav_info and nav_info.nav_type in ('ds_al_fine', 'dc_al_fine'):
            # --- 情況 A: D.S. 或 D.C. al Fine 導航結構 ---
            segno_m = nav_info.segno_m
            fine_m = nav_info.fine_m
            jump_m = nav_info.jump_m

            refrain_notes = [n for n in self.notes if n.measure_number and segno_m <= n.measure_number <= fine_m]
            verse_body_notes = [n for n in self.notes if n.measure_number and fine_m < n.measure_number <= jump_m]
            outro_notes = [n for n in self.notes if n.measure_number and n.measure_number > jump_m]

            current_time_s = 0.0
            for idx, v in enumerate(verse_order):
                seg_start_s = current_time_s

                # Intro Refrain (第一段開頭若有副歌先唱一次)
                if idx == 0 and refrain_notes:
                    for n in refrain_notes:
                        expanded_notes.append(ExpandedNote(n.pitch, n.quarter_length, n.lyrics.get(1, ""), n.measure_number, 1, True))
                        current_time_s += n.quarter_length * self.second_per_beat

                # 主歌
                for n in verse_body_notes:
                    expanded_notes.append(ExpandedNote(n.pitch, n.quarter_length, n.lyrics.get(v, ""), n.measure_number, v, False))
                    current_time_s += n.quarter_length * self.second_per_beat

                # 回副歌唱至 Fine
                for n in refrain_notes:
                    expanded_notes.append(ExpandedNote(n.pitch, n.quarter_length, n.lyrics.get(1, ""), n.measure_number, 1, True))
                    current_time_s += n.quarter_length * self.second_per_beat

                # 最後一段後的 Outro
                if idx == len(verse_order) - 1 and outro_notes:
                    for n in outro_notes:
                        expanded_notes.append(ExpandedNote(n.pitch, n.quarter_length, n.lyrics.get(v, ""), n.measure_number, v, False))
                        current_time_s += n.quarter_length * self.second_per_beat

                seg_end_s = current_time_s
                segments.append(Segment(
                    seg=idx + 1,
                    start_s=seg_start_s,
                    end_s=seg_end_s,
                    start_ms=int(round(seg_start_s * 1000)),
                    end_ms=int(round(seg_end_s * 1000)),
                    lyrics_verse=v
                ))

        elif complex_found or not any(mk.direction == 'backward' for mk in markers):
            # --- 情況 B: 複雜未支援結構 或 無反覆小節線 ---
            total_qlen = sum(n.quarter_length for n in self.notes)
            total_s = total_qlen * self.second_per_beat * max(1, len(verse_order))
            segments.append(Segment(
                seg=1, start_s=0.0, end_s=total_s, start_ms=0,
                end_ms=int(round(total_s * 1000)), lyrics_verse=verse_order[0]
            ))
            for v in verse_order:
                for n in self.notes:
                    expanded_notes.append(ExpandedNote(n.pitch, n.quarter_length, n.lyrics.get(v, ""), n.measure_number, v, False))

        else:
            # --- 情況 C: 標準反覆小節線 (:||) ---
            backward_measures = [mk.measure for mk in markers if mk.direction == 'backward']
            repeat_end_m = max(backward_measures) if backward_measures else None
            current_time_s = 0.0

            for idx, v in enumerate(verse_order):
                is_last_verse = (idx == len(verse_order) - 1)
                st = current_time_s

                for n in self.notes:
                    if (not is_last_verse) and (repeat_end_m is not None) and (n.measure_number is not None) and (n.measure_number > repeat_end_m):
                        continue
                    expanded_notes.append(ExpandedNote(n.pitch, n.quarter_length, n.lyrics.get(v, ""), n.measure_number, v, False))
                    current_time_s += n.quarter_length * self.second_per_beat

                en = current_time_s
                segments.append(Segment(
                    seg=idx + 1, start_s=st, end_s=en,
                    start_ms=int(round(st * 1000)), end_ms=int(round(en * 1000)),
                    lyrics_verse=v
                ))

        return segments, expanded_notes


# ==============================================================================
# 5. 輸出層 (Exporters)
# ==============================================================================

class MidiExporter:
    TICKS_PER_BEAT = 480

    @classmethod
    def _calc_time_sig(cls, beats_qlen: float, default_den: int) -> Tuple[int, int]:
        """依據四分音符時值換算拍號之分子與分母"""
        # 優先嘗試原曲的分母
        num = round(beats_qlen * (default_den / 4.0))
        if num >= 1 and abs(num / (default_den / 4.0) - beats_qlen) < 1e-3:
            return int(num), default_den
        # 嘗試分母為 4
        num = round(beats_qlen)
        if num >= 1 and abs(num - beats_qlen) < 1e-3:
            return int(num), 4
        # 嘗試分母為 8
        num = round(beats_qlen * 2)
        if num >= 1 and abs(num / 2.0 - beats_qlen) < 1e-3:
            return int(num), 8
        # 降級分母為 16
        num = round(beats_qlen * 4)
        return max(1, int(num)), 16

    @classmethod
    def export(cls, notes: List[ExpandedNote], metadata: ScoreMetadata, output_path: str):
        mid = MidiFile(charset='utf-8')
        track = MidiTrack()
        mid.tracks.append(track)

        events: List[Tuple[int, int, Any]] = []

        # 1. 速度設定 (BPM)
        midi_tempo = int(60000000 / metadata.bpm)
        events.append((0, 0, MetaMessage('set_tempo', tempo=midi_tempo)))

        # 2. 拍號設定
        has_pickup = metadata.pickup_beats > 0
        pickup_ticks = int(round(metadata.pickup_beats * cls.TICKS_PER_BEAT)) if has_pickup else 0

        # 計算全曲總 ticks 與最後一小節開始的時間點 (ticks)
        total_ticks = sum(int(round(n.quarter_length * cls.TICKS_PER_BEAT)) for n in notes)
        has_short_last_bar = metadata.last_bar_beats > 0
        if has_short_last_bar:
            last_bar_start_ticks = total_ticks - int(round(metadata.last_bar_beats * cls.TICKS_PER_BEAT))
        else:
            last_bar_start_ticks = -1

        # 寫入開頭拍號 (弱起小節拍號 如 1/4，或主拍號 如 3/4)
        if has_pickup:
            p_num, p_den = cls._calc_time_sig(metadata.pickup_beats, metadata.time_signature_den)
            events.append((0, 0, MetaMessage('time_signature', numerator=p_num, denominator=p_den)))
            # 弱起結束後切回主拍號 (例如第 1 拍結束時切為 3/4)
            events.append((
                pickup_ticks,
                0,
                MetaMessage(
                    'time_signature',
                    numerator=metadata.time_signature_num,
                    denominator=metadata.time_signature_den,
                ),
            ))
        else:
            events.append((
                0,
                0,
                MetaMessage(
                    'time_signature',
                    numerator=metadata.time_signature_num,
                    denominator=metadata.time_signature_den,
                ),
            ))

        # 結尾不完整小節切換為結尾拍號 (例如 2/4)
        if has_short_last_bar and last_bar_start_ticks > 0:
            last_num, last_den = cls._calc_time_sig(metadata.last_bar_beats, metadata.time_signature_den)
            events.append((
                last_bar_start_ticks,
                0,
                MetaMessage('time_signature', numerator=last_num, denominator=last_den),
            ))

        # 3. 收集所有音符與歌詞事件
        current_ticks = 0
        for n in notes:
            duration_ticks = int(round(n.quarter_length * cls.TICKS_PER_BEAT))
            if not n.is_rest:
                if n.lyric_text:
                    events.append((current_ticks, 2, MetaMessage('lyrics', text=n.lyric_text)))
                events.append((current_ticks, 3, Message('note_on', note=n.pitch, velocity=64)))
                events.append((current_ticks + duration_ticks, 1, Message('note_off', note=n.pitch, velocity=64)))
            current_ticks += duration_ticks

        # 4. 結尾軌道標記，確保音軌總長度完整保留 (包含結尾若有休止符)
        events.append((total_ticks, 99, MetaMessage('end_of_track')))

        # 5. 排序事件並轉為 MIDI 相對 delta-time
        # 優先順序: (tick 小到大, 優先級: 0=Meta/TS/Tempo, 1=note_off, 2=lyrics, 3=note_on, 99=end_of_track)
        events.sort(key=lambda x: (x[0], x[1]))

        last_tick = 0
        for tick, _, msg in events:
            delta = tick - last_tick
            msg.time = delta
            track.append(msg)
            last_tick = tick

        mid.save(output_path)
        
class JsonReportExporter:
    """輸出包含 Repeat Markers 與 Segments 分段資訊的 JSON 檔"""

    @staticmethod
    def export(source_xml: str, midi_path: str, metadata: ScoreMetadata,
               markers: List[RepeatMarker], complex_found: bool,
               segments: List[Segment], output_path: str):
        payload = {
            "source": os.path.basename(source_xml),
            "midi": os.path.basename(midi_path),
            "bpm": metadata.bpm,
            "ticks_per_beat": MidiExporter.TICKS_PER_BEAT,
            "second_per_beat": round(metadata.second_per_beat, 3),
            "repeat_markers": [m.to_dict() for m in markers],
            "complex_repeat_present": complex_found,
            "segments": [s.to_dict() for s in segments],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 6. 外觀服務核心 (Facade Service)
# ==============================================================================

class MusicXmlToMidiService:
    """協調整合解析、分析、展開與輸出的主要服務入口"""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.parser = MusicXmlParser(xml_path)

    def get_track_names(self) -> List[str]:
        return list(self.parser.get_parts().keys())

    def convert(self, part_name: Optional[str] = None, show_popup: bool = True):
        try:
            parts = self.parser.get_parts()
            if not parts:
                raise ValueError("樂譜中找不到任何有效聲部！")

            # 若未指定聲部名稱，預設選取第一個
            selected_part = parts[part_name] if part_name else next(iter(parts.values()))

            # 1. 提取詮釋資料與音符
            metadata = self.parser.extract_metadata(selected_part)
            notes = self.parser.parse_note_events(selected_part)

            # 2. 結構導航與反覆分析
            analyzer = RepeatAnalyzer(selected_part, metadata.second_per_beat)
            markers, complex_found, nav_info = analyzer.analyze()

            if complex_found:
                print("偵測到未支援之複雜反覆結構 (例如 Coda 標記等)，此版不分割，整首送往模型。")
            elif nav_info:
                print(f"偵測到導航結構 ({nav_info.nav_type})，已自動展開為標準演出順序。")

            # 3. 展開順序
            expander = PlaybackExpander(notes, metadata.second_per_beat)
            segments, expanded_notes = expander.expand(markers, complex_found, nav_info)

            # 4. 輸出檔案
            base_path, _ = os.path.splitext(self.xml_path)
            midi_path = f"{base_path}_SoulX.mid"
            json_path = f"{base_path}_SoulX.json"

            MidiExporter.export(expanded_notes, metadata, midi_path)
            JsonReportExporter.export(self.xml_path, midi_path, metadata, markers, complex_found, segments, json_path)

            print(f"🎉 轉換成功！\nMIDI: {midi_path}\nJSON: {json_path}")

            if show_popup:
                root_msg = tk.Tk()
                root_msg.withdraw()
                messagebox.showinfo(
                    "成功",
                    f"🎉 MIDI 轉換完成！\n已儲存至：\n{midi_path}\n\nRepeat 對應檔：\n{json_path}"
                )
                root_msg.destroy()

        except Exception as e:
            print(f"轉換失敗: {e}")
            if show_popup:
                root_err = tk.Tk()
                root_err.withdraw()
                messagebox.showerror("轉換失敗", f"發生錯誤：\n{str(e)}")
                root_err.destroy()


# ==============================================================================
# 7. UI 與 CLI 介面層
# ==============================================================================

class ConverterAppUI:
    """Tkinter 聲部選取 UI 介面"""

    def __init__(self, service: MusicXmlToMidiService):
        self.service = service
        self.root = tk.Tk()
        self._setup_window()

    def _setup_window(self):
        self.root.title("SoulX-Singer MIDI 轉換器")
        self.root.geometry("400x180")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        # 視窗置中
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_reqheight()) // 2
        self.root.geometry(f"+{x}+{y}")

        track_names = self.service.get_track_names()

        label = ttk.Label(
            self.root,
            text=f"已載入: {os.path.basename(self.service.xml_path)}\n請選擇【主旋律歌詞軌】:",
            font=("Arial", 10)
        )
        label.pack(pady=15)

        self.combo = ttk.Combobox(self.root, values=track_names, state="readonly", width=35)
        self.combo.pack(pady=5)
        if track_names:
            self.combo.current(0)

        btn_confirm = ttk.Button(self.root, text="開始轉換", command=self._on_confirm)
        btn_confirm.pack(pady=15)

    def _on_confirm(self):
        selected_part_name = self.combo.get()
        self.root.destroy()
        self.service.convert(selected_part_name, show_popup=True)

    def run(self):
        self.root.mainloop()


# ==============================================================================
# 程式進入點
# ==============================================================================

def main():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # CLI 模式
        file_path = sys.argv[1]
        service = MusicXmlToMidiService(file_path)
        service.convert(part_name=None, show_popup=False)
    else:
        # GUI 檔案選擇器模式
        root_init = tk.Tk()
        root_init.withdraw()

        file_path = filedialog.askopenfilename(
            title="請選取從 MuseScore 導出的 MusicXML 檔案",
            filetypes=[("MusicXML 檔案", "*.musicxml *.xml"), ("所有檔案", "*.*")]
        )
        root_init.destroy()

        if file_path:
            service = MusicXmlToMidiService(file_path)
            app = ConverterAppUI(service)
            app.run()
        else:
            print("未選擇任何檔案，程式結束。")


if __name__ == "__main__":
    main()