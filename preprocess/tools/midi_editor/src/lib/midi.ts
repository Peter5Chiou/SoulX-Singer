import { Midi } from '@tonejs/midi'
import { parseMidi, writeMidi } from 'midi-file'
import type { MidiData, MidiEvent } from 'midi-file'
import type { NoteEvent, ProjectSnapshot, TimeSignature } from '../types'

const DEFAULT_SIGNATURE: TimeSignature = [4, 4]

export type MidiChannelInfo = {
  key: string // channel number as string (selection value)
  channel: number
  name: string
  notes: number
  low: number | null
  high: number | null
}

const GM_PROGRAM_NAMES = [
  'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
  'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavi',
  'Celesta', 'Glockenspiel', 'Music Box', 'Vibraphone',
  'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer',
  'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ',
  'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion',
  'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (jazz)', 'Electric Guitar (clean)',
  'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Guitar Harmonics',
  'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass',
  'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2',
  'Violin', 'Viola', 'Cello', 'Contrabass',
  'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani',
  'String Ensemble 1', 'String Ensemble 2', 'Synth Strings 1', 'Synth Strings 2',
  'Choir Aahs', 'Voice Oohs', 'Synth Voice', 'Orchestra Hit',
  'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet',
  'French Horn', 'Brass Section', 'Synth Brass 1', 'Synth Brass 2',
  'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax',
  'Oboe', 'English Horn', 'Bassoon', 'Clarinet',
  'Piccolo', 'Flute', 'Recorder', 'Pan Flute',
  'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina',
  'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)',
  'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)',
  'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)',
  'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)',
  'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)',
  'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)',
  'Sitar', 'Banjo', 'Shamisen', 'Koto',
  'Kalimba', 'Bag pipe', 'Fiddle', 'Shanai',
  'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock',
  'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal',
  'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet',
  'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot',
]

export function midiNoteName(note: number): string {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  return `${names[note % 12]}${Math.floor(note / 12) - 1}`
}

/**
 * Filter a raw MIDI buffer to keep only note/controller events on the given channels.
 * Channelless meta events (tempo, time signature, lyrics, etc.) are always preserved.
 */
function filterMidiByChannels(buffer: ArrayBuffer, channels: number[]): ArrayBuffer {
  const midiData = parseMidi(new Uint8Array(buffer))
  const keep = new Set<number>(channels)
  const tracks = midiData.tracks.map((track) =>
    track.filter((ev) => {
      switch (ev.type) {
        case 'noteOn':
        case 'noteOff':
        case 'programChange':
        case 'pitchBend':
        case 'controller':
        case 'channelAftertouch':
        case 'noteAftertouch':
          return keep.has(ev.channel)
        default:
          return true
      }
    }),
  )
  const bytes = writeMidi({ header: midiData.header, tracks })
  return new Uint8Array(bytes).buffer
}

/**
 * Inspect a MIDI buffer and report one entry per channel that contains note events
 * (note count, pitch range, instrument name). Channels are the natural grouping so
 * importing a multi-channel MIDI lets the user pick which channel(s) to use.
 */
export function inspectMidiChannelsBuffer(buffer: ArrayBuffer): MidiChannelInfo[] {
  const midiData = parseMidi(new Uint8Array(buffer))
  const perChannel = new Map<
    number,
    { count: number; low: number; high: number; program?: number }
  >()
  let trackName = ''
  for (const track of midiData.tracks) {
    for (const ev of track) {
      if (ev.type === 'trackName' && ev.text) {
        trackName = ev.text
      } else if (ev.type === 'programChange') {
        const cur = perChannel.get(ev.channel) || { count: 0, low: Infinity, high: -Infinity }
        cur.program = ev.programNumber
        perChannel.set(ev.channel, cur)
      } else if (ev.type === 'noteOn' && ev.velocity > 0) {
        const cur = perChannel.get(ev.channel) || { count: 0, low: Infinity, high: -Infinity }
        cur.count += 1
        cur.low = Math.min(cur.low, ev.noteNumber)
        cur.high = Math.max(cur.high, ev.noteNumber)
        perChannel.set(ev.channel, cur)
      }
    }
  }
  const infos: MidiChannelInfo[] = []
  for (const [channel, info] of [...perChannel.entries()].sort((a, b) => a[0] - b[0])) {
    if (info.count <= 0) continue
    const name =
      info.program !== undefined
        ? GM_PROGRAM_NAMES[info.program] || `Program ${info.program}`
        : trackName || ''
    infos.push({
      key: String(channel),
      channel,
      name,
      notes: info.count,
      low: info.low === Infinity ? null : info.low,
      high: info.high === -Infinity ? null : info.high,
    })
  }
  return infos
}

export function inspectMidiChannels(file: File): Promise<MidiChannelInfo[]> {
  return file.arrayBuffer().then(inspectMidiChannelsBuffer)
}

// Decode UTF-8 byte string (latin1 encoded) to proper Unicode string
// This matches: text.encode("latin1").decode("utf-8") in Python
function decodeUtf8ByteString(byteString: string): string {
  try {
    const bytes = new Uint8Array(byteString.length)
    for (let i = 0; i < byteString.length; i++) {
      bytes[i] = byteString.charCodeAt(i)
    }
    return new TextDecoder('utf-8').decode(bytes)
  } catch {
    return byteString
  }
}

// Encode Unicode string to UTF-8 byte string (latin1 encoding)
// This matches: text.encode("utf-8").decode("latin1") in Python
function encodeUtf8ByteString(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let output = ''
  bytes.forEach((b) => {
    output += String.fromCharCode(b)
  })
  return output
}

export async function importMidiFile(
  file: File,
  selectedChannels?: number[],
): Promise<ProjectSnapshot> {
  const buffer = await file.arrayBuffer()
  return parseMidiBuffer(buffer, selectedChannels)
}

export async function parseMidiBuffer(
  buffer: ArrayBuffer,
  selectedChannels?: number[],
): Promise<ProjectSnapshot> {
  if (selectedChannels && selectedChannels.length > 0) {
    buffer = filterMidiByChannels(buffer, selectedChannels)
  }
  const midi = new Midi(buffer)
  const tempo = midi.header.tempos[0]?.bpm ?? 120
  const timeSignature = (midi.header.timeSignatures[0]?.timeSignature as TimeSignature | undefined) ?? DEFAULT_SIGNATURE
  const ppq = midi.header.ppq
  
  // Merge notes from all tracks and sort by ticks then by midi (for stable ordering)
  const allNotes = midi.tracks
    .flatMap(t => t.notes)
    .sort((a, b) => a.ticks - b.ticks || a.midi - b.midi)
  
  // Parse lyrics directly from the raw MIDI buffer using midi-file,
  // because @tonejs/midi's header.meta loses correct tick positions for
  // lyrics in Format 1 MIDI files (all lyrics collapse to tick 0).
  // We accumulate absolute ticks across deltaTime events.
  const rawMidi = parseMidi(new Uint8Array(buffer))
  interface RawLyricEvent {
    tick: number
    text: string
  }
  const lyricEvents: RawLyricEvent[] = []
  for (const track of rawMidi.tracks) {
    let absoluteTick = 0
    for (const ev of track) {
      absoluteTick += ev.deltaTime
      if (ev.type === 'lyrics') {
        lyricEvents.push({ tick: absoluteTick, text: decodeUtf8ByteString(ev.text) })
      }
    }
  }
  // Sort lyrics by tick position
  lyricEvents.sort((a, b) => a.tick - b.tick)
  
  // Match lyrics to notes by tick position
  // Each lyric should be consumed by exactly one note at the same tick.
  // We use a sequential dual-pointer approach: both lyrics and notes are
  // sorted by tick. We walk through lyrics, matching each to the next
  // available note at the same (or nearest) tick.
  const notes: NoteEvent[] = allNotes.map((note, index) => {
    const beat = note.ticks / ppq
    const durationBeats = note.durationTicks / ppq
    
    let lyric = ''
    
    // Exact tick match: consume the first unmatched lyric at this tick
    for (let i = 0; i < lyricEvents.length; i++) {
      const lyr = lyricEvents[i]
      if (lyr.tick === note.ticks) {
        lyric = lyr.text
        lyricEvents.splice(i, 1)
        break
      }
    }
    
    // If no exact match, try nearby ticks (within small tolerance)
    if (!lyric) {
      const tolerance = ppq / 100 // Very small tolerance
      for (let i = 0; i < lyricEvents.length; i++) {
        const lyr = lyricEvents[i]
        if (Math.abs(lyr.tick - note.ticks) <= tolerance) {
          lyric = lyr.text
          lyricEvents.splice(i, 1)
          break
        }
      }
    }

    return {
      id: `${index}-${note.midi}-${Math.round(note.ticks)}`,
      midi: note.midi,
      start: beat,
      duration: Math.max(durationBeats, 0.0625),
      velocity: note.velocity,
      lyric,
    }
  })

  return { tempo, timeSignature, notes, ppq }
}

// Used to add absoluteTime property for sorting
type WithAbsoluteTime<T> = T & { absoluteTime: number }

export function exportMidi(snapshot: ProjectSnapshot): Blob {
  const ppq = snapshot.ppq ?? 480  // Use original ppq if available, otherwise default to 480
  const microsecondsPerBeat = Math.round(60000000 / snapshot.tempo)  // Convert BPM to microseconds per beat

  // Sort notes by start time, then by midi for stable ordering
  const sortedNotes = [...snapshot.notes].sort((a, b) => a.start - b.start || a.midi - b.midi)

  // Build events for a single track containing both lyrics and notes
  // Event order at same tick: note_off (0) < lyrics (1) < note_on (2)
  // This matches meta.py's tg2midi implementation
  const events: Array<WithAbsoluteTime<MidiEvent>> = []

  // Add all note events and their corresponding lyrics
  sortedNotes.forEach((note) => {
    const startTicks = Math.round(note.start * ppq)
    const endTicks = Math.round((note.start + note.duration) * ppq)
    const velocity = Math.round(note.velocity * 127)

    // Add lyric event at the same tick as note_on (but will be sorted before it)
    const lyricText = note.lyric ?? ''
    const encodedLyric = encodeUtf8ByteString(lyricText)
    
    // Lyric event - sort key 1 (after note_off, before note_on)
    events.push({
      absoluteTime: startTicks,
      deltaTime: 0,
      meta: true,
      type: 'lyrics',
      text: encodedLyric,
      _sortKey: 1,
    } as WithAbsoluteTime<MidiEvent> & { _sortKey: number })

    // Note on event - sort key 2 (after lyrics)
    events.push({
      absoluteTime: startTicks,
      deltaTime: 0,
      type: 'noteOn',
      channel: 0,
      noteNumber: note.midi,
      velocity: velocity,
      _sortKey: 2,
    } as WithAbsoluteTime<MidiEvent> & { _sortKey: number })

    // Note off event - sort key 0 (before everything at same tick)
    events.push({
      absoluteTime: endTicks,
      deltaTime: 0,
      type: 'noteOff',
      channel: 0,
      noteNumber: note.midi,
      velocity: 0,
      _sortKey: 0,
    } as WithAbsoluteTime<MidiEvent> & { _sortKey: number })
  })

  // Sort events by absoluteTime, then by _sortKey
  events.sort((a, b) => {
    const aKey = (a as { _sortKey?: number })._sortKey ?? 1
    const bKey = (b as { _sortKey?: number })._sortKey ?? 1
    return a.absoluteTime - b.absoluteTime || aKey - bKey
  })

  // Convert absolute time to delta time
  let lastTick = 0
  events.forEach(event => {
    event.deltaTime = event.absoluteTime - lastTick
    lastTick = event.absoluteTime
    delete (event as { absoluteTime?: number }).absoluteTime
    delete (event as { _sortKey?: number })._sortKey
  })

  // Build the MIDI track with header events
  const track: MidiEvent[] = [
    // Set tempo
    {
      deltaTime: 0,
      meta: true,
      type: 'setTempo',
      microsecondsPerBeat: microsecondsPerBeat,
    },
    // Time signature
    {
      deltaTime: 0,
      meta: true,
      type: 'timeSignature',
      numerator: snapshot.timeSignature[0],
      denominator: snapshot.timeSignature[1],
      metronome: 24,
      thirtyseconds: 8,
    },
    // All note and lyric events
    ...events,
    // End of track
    {
      deltaTime: 0,
      meta: true,
      type: 'endOfTrack',
    },
  ]

  // Build MIDI data structure
  const midiData: MidiData = {
    header: {
      format: 0,  // Single track format (type 0)
      numTracks: 1,
      ticksPerBeat: ppq,
    },
    tracks: [track],
  }

  const bytes = writeMidi(midiData)
  return new Blob([new Uint8Array(bytes)], { type: 'audio/midi' })
}
