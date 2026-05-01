"""Generate FCP XML (Final Cut Pro 7 XML) for Premiere Pro import."""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from pathlib import Path


def _frames_to_timecode(frames: int, fps: float) -> str:
    """Convert frame count to SMPTE timecode string (HH:MM:SS:FF)."""
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frame_remainder = int(frames % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame_remainder:02d}"


def _timecode_to_frames(timecode: str, fps: float) -> int:
    """Convert SMPTE timecode to frame count."""
    parts = timecode.split(":")
    hours, minutes, seconds, frames = map(int, parts)
    return int((hours * 3600 + minutes * 60 + seconds) * fps + frames)


def _seconds_to_timecode(seconds: float, fps: float) -> str:
    """Convert seconds to SMPTE timecode."""
    return _frames_to_timecode(int(seconds * fps), fps)


def generate_premiere_xml(
    storyboard: dict,
    output_xml_path: str,
    video_path: str,
    project_name: str = "MovieRecap",
) -> str:
    """
    Generate FCP XML 7 file for import into Premiere Pro.
    
    Returns path to the generated XML file.
    """
    fps = storyboard.get("fps", 30)
    scenes = storyboard.get("scenes", [])
    
    # Calculate total duration
    total_frames = sum(s.get("displayFrames", s.get("durationInFrames", 30)) for s in scenes)
    total_duration_tc = _frames_to_timecode(total_frames, fps)
    
    # Create root element
    xmeml = ET.Element("xmeml", {"version": "7"})
    sequence = ET.SubElement(xmeml, "sequence")
    
    # Sequence metadata
    ET.SubElement(sequence, "name").text = project_name
    ET.SubElement(sequence, "duration").text = str(total_frames)
    
    # Rate (fps)
    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(int(fps))
    ET.SubElement(rate, "ntsc").text = "FALSE" if fps in [24, 25, 30] else "TRUE"
    
    # Timecode
    tc = ET.SubElement(sequence, "timecode")
    tc_rate = ET.SubElement(tc, "rate")
    ET.SubElement(tc_rate, "timebase").text = str(int(fps))
    ET.SubElement(tc_rate, "ntsc").text = "FALSE" if fps in [24, 25, 30] else "TRUE"
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"
    
    # Media
    media = ET.SubElement(sequence, "media")
    
    # Video track
    video = ET.SubElement(media, "video")
    video_format = ET.SubElement(video, "format")
    video_sample = ET.SubElement(video_format, "samplecharacteristics")
    ET.SubElement(video_sample, "width").text = "1920"
    ET.SubElement(video_sample, "height").text = "1080"
    ET.SubElement(video_sample, "pixelaspectratio").text = "square"
    
    # Video track 1
    vtrack = ET.SubElement(video, "track")
    
    # Add video clips for each scene
    current_frame = 0
    for i, scene in enumerate(scenes):
        start_sec = scene.get("startSec", 0)
        end_sec = scene.get("endSec", start_sec + 1)
        display_frames = scene.get("displayFrames", scene.get("durationInFrames", 30))
        
        # Source timing in original video
        src_start_frame = int(start_sec * fps)
        src_end_frame = src_start_frame + display_frames
        
        # Timeline timing
        timeline_in = current_frame
        timeline_out = current_frame + display_frames
        
        clipitem = ET.SubElement(vtrack, "clipitem", {"id": f"clipitem-{i+1}"})
        ET.SubElement(clipitem, "name").text = f"Scene {i+1}"
        ET.SubElement(clipitem, "enabled").text = "TRUE"
        ET.SubElement(clipitem, "duration").text = str(display_frames)
        ET.SubElement(clipitem, "start").text = str(timeline_in)
        ET.SubElement(clipitem, "end").text = str(timeline_out)
        ET.SubElement(clipitem, "in").text = str(src_start_frame)
        ET.SubElement(clipitem, "out").text = str(src_end_frame)
        
        # File reference
        file_elem = ET.SubElement(clipitem, "file", {"id": "file-1"})
        ET.SubElement(file_elem, "name").text = Path(video_path).name
        ET.SubElement(file_elem, "pathurl").text = f"file://localhost{os.path.abspath(video_path)}"
        
        # File rate
        file_rate = ET.SubElement(file_elem, "rate")
        ET.SubElement(file_rate, "timebase").text = str(int(fps))
        ET.SubElement(file_rate, "ntsc").text = "FALSE" if fps in [24, 25, 30] else "TRUE"
        
        # Duration
        ET.SubElement(file_elem, "duration").text = str(int(os.path.getsize(video_path) / 1000))  # Approximate
        
        # Timecode
        file_tc = ET.SubElement(file_elem, "timecode")
        file_tc_rate = ET.SubElement(file_tc, "rate")
        ET.SubElement(file_tc_rate, "timebase").text = str(int(fps))
        ET.SubElement(file_tc_rate, "ntsc").text = "FALSE" if fps in [24, 25, 30] else "TRUE"
        ET.SubElement(file_tc, "string").text = "00:00:00:00"
        
        current_frame += display_frames
    
    # Audio track (voiceovers)
    audio = ET.SubElement(media, "audio")
    
    # Audio format
    audio_format = ET.SubElement(audio, "format")
    audio_sample = ET.SubElement(audio_format, "samplecharacteristics")
    ET.SubElement(audio_sample, "depth").text = "16"
    ET.SubElement(audio_sample, "samplerate").text = "48000"
    
    # Audio track 1 (Voiceover)
    atrack = ET.SubElement(audio, "track")
    ET.SubElement(atrack, "enabled").text = "TRUE"
    ET.SubElement(atrack, "locked").text = "FALSE"
    
    # Add audio clips
    current_frame = 0
    for i, scene in enumerate(scenes):
        display_frames = scene.get("displayFrames", scene.get("durationInFrames", 30))
        voiceover_path = scene.get("voiceoverPath")
        
        if voiceover_path:
            timeline_in = current_frame
            timeline_out = current_frame + display_frames
            
            clipitem = ET.SubElement(atrack, "clipitem", {"id": f"audioclip-{i+1}"})
            ET.SubElement(clipitem, "name").text = f"Voiceover {i+1}"
            ET.SubElement(clipitem, "enabled").text = "TRUE"
            ET.SubElement(clipitem, "duration").text = str(display_frames)
            ET.SubElement(clipitem, "start").text = str(timeline_in)
            ET.SubElement(clipitem, "end").text = str(timeline_out)
            ET.SubElement(clipitem, "in").text = "0"
            ET.SubElement(clipitem, "out").text = str(display_frames)
            
            # File reference
            vo_full_path = os.path.join(
                os.path.dirname(os.path.dirname(voiceover_path)),
                "voiceover",
                os.path.basename(voiceover_path)
            )
            if not os.path.exists(vo_full_path):
                vo_full_path = voiceover_path
            
            file_elem = ET.SubElement(clipitem, "file", {"id": f"audiofile-{i+1}"})
            ET.SubElement(file_elem, "name").text = os.path.basename(voiceover_path)
            ET.SubElement(file_elem, "pathurl").text = f"file://localhost{os.path.abspath(vo_full_path)}"
            
            file_rate = ET.SubElement(file_elem, "rate")
            ET.SubElement(file_rate, "timebase").text = str(int(fps))
            ET.SubElement(file_rate, "ntsc").text = "FALSE" if fps in [24, 25, 30] else "TRUE"
            ET.SubElement(file_elem, "duration").text = str(display_frames)
            
            # Sourcetrack
            sourcetrack = ET.SubElement(clipitem, "sourcetrack")
            ET.SubElement(sourcetrack, "mediatype").text = "audio"
            ET.SubElement(sourcetrack, "trackindex").text = "1"
        
        current_frame += display_frames
    
    # Write XML with proper formatting
    xml_string = ET.tostring(xmeml, encoding="unicode")
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    # Remove empty lines
    pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip()])
    
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(pretty_xml)
    
    return output_xml_path


def generate_premiere_edl(
    storyboard: dict,
    output_edl_path: str,
    project_name: str = "MovieRecap",
) -> str:
    """
    Generate CMX3600 EDL file for Premiere Pro.
    Simpler format, good for rough cuts.
    """
    fps = storyboard.get("fps", 30)
    scenes = storyboard.get("scenes", [])
    
    lines = []
    lines.append(f"TITLE: {project_name}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")
    
    current_frame = 0
    for i, scene in enumerate(scenes):
        start_sec = scene.get("startSec", 0)
        display_frames = scene.get("displayFrames", scene.get("durationInFrames", 30))
        end_sec = start_sec + (display_frames / fps)
        
        # Source timing
        src_in = _seconds_to_timecode(start_sec, fps)
        src_out = _seconds_to_timecode(end_sec, fps)
        
        # Timeline timing
        timeline_in = _frames_to_timecode(current_frame, fps)
        timeline_out = _frames_to_timecode(current_frame + display_frames, fps)
        
        # EDL line: EDIT# REEL_NAME CHANNEL SOURCE_IN SOURCE_OUT TIMELINE_IN TIMELINE_OUT
        edit_num = i + 1
        lines.append(f"{edit_num:03d}  SCENE_{i+1:03d}  V     C        {src_in} {src_out} {timeline_in} {timeline_out}")
        lines.append(f"* Scene {i+1}: {scene.get('povText', '')[:50]}")
        lines.append("")
        
        current_frame += display_frames
    
    with open(output_edl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return output_edl_path
