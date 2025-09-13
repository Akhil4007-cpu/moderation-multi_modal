#!/usr/bin/env python
"""
Weapon Detection Analysis - Extract and analyze weapon detections from video results
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def analyze_weapon_detections(detailed_results_path):
    """
    Extract and analyze all weapon detections from the detailed results
    """
    
    # Load detailed results
    with open(detailed_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    weapon_detections = []
    frames_with_weapons = []
    total_weapons = 0
    
    # Process each frame
    for frame_data in data['frame_results']:
        frame_num = frame_data['frame_number']
        second = frame_data['second']
        frame_file = frame_data['frame_file']
        
        # Check weapon model results
        if 'weapon' in frame_data['model_results']:
            weapon_result = frame_data['model_results']['weapon']
            
            if weapon_result['detected'] and weapon_result['detections']:
                frame_weapons = []
                
                for detection in weapon_result['detections']:
                    weapon_info = {
                        'frame_number': frame_num,
                        'second': second,
                        'frame_file': frame_file,
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox'],
                        'bbox_center': [
                            (detection['bbox'][0] + detection['bbox'][2]) / 2,
                            (detection['bbox'][1] + detection['bbox'][3]) / 2
                        ],
                        'bbox_size': [
                            detection['bbox'][2] - detection['bbox'][0],
                            detection['bbox'][3] - detection['bbox'][1]
                        ]
                    }
                    
                    weapon_detections.append(weapon_info)
                    frame_weapons.append(weapon_info)
                    total_weapons += 1
                
                frames_with_weapons.append({
                    'frame_number': frame_num,
                    'second': second,
                    'frame_file': frame_file,
                    'weapon_count': len(frame_weapons),
                    'weapons': frame_weapons,
                    'max_confidence': max(w['confidence'] for w in frame_weapons),
                    'avg_confidence': sum(w['confidence'] for w in frame_weapons) / len(frame_weapons)
                })
    
    # Sort by confidence
    weapon_detections.sort(key=lambda x: x['confidence'], reverse=True)
    frames_with_weapons.sort(key=lambda x: x['max_confidence'], reverse=True)
    
    # Create analysis report
    analysis = {
        'video_info': {
            'file': data['video_path'],
            'duration': data['video_info']['duration'],
            'total_frames_analyzed': data['total_frames_processed']
        },
        'weapon_analysis': {
            'total_weapon_detections': total_weapons,
            'frames_with_weapons': len(frames_with_weapons),
            'frames_without_weapons': data['total_frames_processed'] - len(frames_with_weapons),
            'weapon_detection_rate': len(frames_with_weapons) / data['total_frames_processed'] * 100,
            'max_confidence': max(w['confidence'] for w in weapon_detections) if weapon_detections else 0,
            'min_confidence': min(w['confidence'] for w in weapon_detections) if weapon_detections else 0,
            'avg_confidence': sum(w['confidence'] for w in weapon_detections) / len(weapon_detections) if weapon_detections else 0,
            'high_confidence_weapons': len([w for w in weapon_detections if w['confidence'] >= 0.7]),
            'medium_confidence_weapons': len([w for w in weapon_detections if 0.5 <= w['confidence'] < 0.7]),
            'low_confidence_weapons': len([w for w in weapon_detections if w['confidence'] < 0.5])
        },
        'timeline': frames_with_weapons,
        'all_detections': weapon_detections,
        'confidence_breakdown': {
            'high_confidence': [w for w in weapon_detections if w['confidence'] >= 0.7],
            'medium_confidence': [w for w in weapon_detections if 0.5 <= w['confidence'] < 0.7],
            'low_confidence': [w for w in weapon_detections if w['confidence'] < 0.5]
        }
    }
    
    return analysis

def print_weapon_analysis(analysis):
    """Print formatted weapon analysis"""
    
    print(f"\n{'='*70}")
    print(f"🔫 WEAPON DETECTION ANALYSIS")
    print(f"{'='*70}")
    
    video_name = Path(analysis['video_info']['file']).name
    print(f"🎥 Video: {video_name}")
    print(f"⏱️  Duration: {analysis['video_info']['duration']:.2f} seconds")
    print(f"🔍 Frames Analyzed: {analysis['video_info']['total_frames_analyzed']}")
    
    wa = analysis['weapon_analysis']
    print(f"\n🚨 WEAPON DETECTION SUMMARY:")
    print(f"   Total Weapons Detected: {wa['total_weapon_detections']}")
    print(f"   Frames with Weapons: {wa['frames_with_weapons']}")
    print(f"   Frames without Weapons: {wa['frames_without_weapons']}")
    print(f"   Detection Rate: {wa['weapon_detection_rate']:.1f}%")
    
    print(f"\n📊 CONFIDENCE STATISTICS:")
    print(f"   Max Confidence: {wa['max_confidence']:.3f}")
    print(f"   Min Confidence: {wa['min_confidence']:.3f}")
    print(f"   Average Confidence: {wa['avg_confidence']:.3f}")
    
    print(f"\n📈 CONFIDENCE BREAKDOWN:")
    print(f"   🔴 High Confidence (≥0.7): {wa['high_confidence_weapons']} weapons")
    print(f"   🟡 Medium Confidence (0.5-0.7): {wa['medium_confidence_weapons']} weapons")
    print(f"   🟢 Low Confidence (<0.5): {wa['low_confidence_weapons']} weapons")
    
    print(f"\n⏰ TIMELINE - FRAMES WITH WEAPONS:")
    for i, frame in enumerate(analysis['timeline'], 1):
        conf_icon = "🔴" if frame['max_confidence'] >= 0.7 else "🟡" if frame['max_confidence'] >= 0.5 else "🟢"
        print(f"   {i}. {conf_icon} Second {frame['second']} (Frame {frame['frame_number']}): {frame['weapon_count']} weapon(s)")
        print(f"      Max Confidence: {frame['max_confidence']:.3f}, Avg: {frame['avg_confidence']:.3f}")
        
        # Show individual weapons in this frame
        for j, weapon in enumerate(frame['weapons'], 1):
            print(f"      Weapon {j}: {weapon['class_name']} (conf: {weapon['confidence']:.3f})")
            print(f"                 Position: ({weapon['bbox_center'][0]:.0f}, {weapon['bbox_center'][1]:.0f})")
    
    print(f"\n🎯 TOP 10 HIGHEST CONFIDENCE DETECTIONS:")
    for i, weapon in enumerate(analysis['all_detections'][:10], 1):
        conf_icon = "🔴" if weapon['confidence'] >= 0.7 else "🟡" if weapon['confidence'] >= 0.5 else "🟢"
        print(f"   {i}. {conf_icon} Frame {weapon['frame_number']} (Second {weapon['second']})")
        print(f"      Class: {weapon['class_name']}")
        print(f"      Confidence: {weapon['confidence']:.3f}")
        print(f"      Position: ({weapon['bbox_center'][0]:.0f}, {weapon['bbox_center'][1]:.0f})")
        print(f"      Size: {weapon['bbox_size'][0]:.0f} x {weapon['bbox_size'][1]:.0f}")
    
    print(f"{'='*70}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python weapon_detection_analysis.py <detailed_results.json>")
        sys.exit(1)
    
    detailed_results_path = Path(sys.argv[1])
    
    if not detailed_results_path.exists():
        print(f"Error: File not found: {detailed_results_path}")
        sys.exit(1)
    
    print(f"🔍 Analyzing weapon detections from: {detailed_results_path}")
    
    try:
        analysis = analyze_weapon_detections(detailed_results_path)
        
        # Print analysis
        print_weapon_analysis(analysis)
        
        # Save detailed weapon analysis
        input_stem = detailed_results_path.stem.replace('_detailed_results', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = detailed_results_path.parent / f"{input_stem}_weapon_analysis_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed weapon analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error analyzing weapon detections: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
