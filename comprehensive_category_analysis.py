#!/usr/bin/env python
"""
Comprehensive Category Analysis - Extract and analyze all model categories from video results
Includes detailed breakdowns for weapon, accident, fight, fire, nsfw, and objects detection
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

def analyze_all_categories(detailed_results_path):
    """
    Extract and analyze all category detections from the detailed results
    """
    
    # Load detailed results
    with open(detailed_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize category analysis structure
    category_analysis = {
        'video_info': {
            'file': data['video_path'],
            'duration': data['video_info']['duration'],
            'total_frames_analyzed': data['total_frames_processed']
        },
        'categories': {}
    }
    
    # Define categories to analyze
    categories = ['weapon', 'accident', 'fight_nano', 'fight_small', 'fire_n', 'fire_s', 'nsfw_cls', 'nsfw_seg', 'objects']
    
    for category in categories:
        category_data = {
            'total_detections': 0,
            'total_classifications': 0,
            'frames_with_activity': 0,
            'frames_without_activity': 0,
            'activity_rate': 0.0,
            'detection_stats': {
                'max_confidence': 0.0,
                'min_confidence': 1.0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0,
                'medium_confidence_count': 0,
                'low_confidence_count': 0
            },
            'classification_stats': {
                'max_probability': 0.0,
                'min_probability': 1.0,
                'avg_probability': 0.0,
                'high_probability_count': 0,
                'medium_probability_count': 0,
                'low_probability_count': 0
            },
            'timeline': [],
            'all_detections': [],
            'all_classifications': [],
            'class_distribution': Counter(),
            'confidence_breakdown': {
                'high_confidence': [],
                'medium_confidence': [],
                'low_confidence': []
            },
            'probability_breakdown': {
                'high_probability': [],
                'medium_probability': [],
                'low_probability': []
            }
        }
        
        frames_with_activity = set()
        
        # Process each frame for this category
        for frame_data in data['frame_results']:
            frame_num = frame_data['frame_number']
            second = frame_data['second']
            frame_file = frame_data['frame_file']
            
            if category in frame_data['model_results']:
                result = frame_data['model_results'][category]
                
                frame_activity = {
                    'frame_number': frame_num,
                    'second': second,
                    'frame_file': frame_file,
                    'detection_count': 0,
                    'classification_count': 0,
                    'detections': [],
                    'classifications': [],
                    'max_confidence': 0.0,
                    'max_probability': 0.0
                }
                
                # Process detections
                if result['detected'] and result['detections']:
                    frames_with_activity.add(frame_num)
                    
                    for detection in result['detections']:
                        conf = detection['confidence']
                        class_name = detection['class_name']
                        
                        detection_info = {
                            'frame_number': frame_num,
                            'second': second,
                            'frame_file': frame_file,
                            'class_name': class_name,
                            'confidence': conf,
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
                        
                        category_data['all_detections'].append(detection_info)
                        frame_activity['detections'].append(detection_info)
                        category_data['total_detections'] += 1
                        category_data['class_distribution'][class_name] += 1
                        
                        # Update confidence stats
                        category_data['detection_stats']['max_confidence'] = max(
                            category_data['detection_stats']['max_confidence'], conf)
                        category_data['detection_stats']['min_confidence'] = min(
                            category_data['detection_stats']['min_confidence'], conf)
                        frame_activity['max_confidence'] = max(frame_activity['max_confidence'], conf)
                        
                        # Categorize by confidence
                        if conf >= 0.7:
                            category_data['detection_stats']['high_confidence_count'] += 1
                            category_data['confidence_breakdown']['high_confidence'].append(detection_info)
                        elif conf >= 0.5:
                            category_data['detection_stats']['medium_confidence_count'] += 1
                            category_data['confidence_breakdown']['medium_confidence'].append(detection_info)
                        else:
                            category_data['detection_stats']['low_confidence_count'] += 1
                            category_data['confidence_breakdown']['low_confidence'].append(detection_info)
                
                # Process classifications
                if result['detected'] and result['classifications']:
                    frames_with_activity.add(frame_num)
                    
                    for classification in result['classifications']:
                        prob = classification['probability']
                        class_name = classification['class_name']
                        
                        classification_info = {
                            'frame_number': frame_num,
                            'second': second,
                            'frame_file': frame_file,
                            'class_name': class_name,
                            'probability': prob
                        }
                        
                        category_data['all_classifications'].append(classification_info)
                        frame_activity['classifications'].append(classification_info)
                        category_data['total_classifications'] += 1
                        category_data['class_distribution'][class_name] += 1
                        
                        # Update probability stats
                        category_data['classification_stats']['max_probability'] = max(
                            category_data['classification_stats']['max_probability'], prob)
                        category_data['classification_stats']['min_probability'] = min(
                            category_data['classification_stats']['min_probability'], prob)
                        frame_activity['max_probability'] = max(frame_activity['max_probability'], prob)
                        
                        # Categorize by probability
                        if prob >= 0.7:
                            category_data['classification_stats']['high_probability_count'] += 1
                            category_data['probability_breakdown']['high_probability'].append(classification_info)
                        elif prob >= 0.5:
                            category_data['classification_stats']['medium_probability_count'] += 1
                            category_data['probability_breakdown']['medium_probability'].append(classification_info)
                        else:
                            category_data['classification_stats']['low_probability_count'] += 1
                            category_data['probability_breakdown']['low_probability'].append(classification_info)
                
                # Add frame to timeline if there was activity
                frame_activity['detection_count'] = len(frame_activity['detections'])
                frame_activity['classification_count'] = len(frame_activity['classifications'])
                
                if frame_activity['detection_count'] > 0 or frame_activity['classification_count'] > 0:
                    category_data['timeline'].append(frame_activity)
        
        # Calculate final stats
        category_data['frames_with_activity'] = len(frames_with_activity)
        category_data['frames_without_activity'] = data['total_frames_processed'] - len(frames_with_activity)
        category_data['activity_rate'] = len(frames_with_activity) / data['total_frames_processed'] * 100
        
        # Calculate averages
        if category_data['total_detections'] > 0:
            total_conf = sum(d['confidence'] for d in category_data['all_detections'])
            category_data['detection_stats']['avg_confidence'] = total_conf / category_data['total_detections']
        else:
            category_data['detection_stats']['min_confidence'] = 0.0
            
        if category_data['total_classifications'] > 0:
            total_prob = sum(c['probability'] for c in category_data['all_classifications'])
            category_data['classification_stats']['avg_probability'] = total_prob / category_data['total_classifications']
        else:
            category_data['classification_stats']['min_probability'] = 0.0
        
        # Sort data
        category_data['all_detections'].sort(key=lambda x: x['confidence'], reverse=True)
        category_data['all_classifications'].sort(key=lambda x: x['probability'], reverse=True)
        category_data['timeline'].sort(key=lambda x: max(x['max_confidence'], x['max_probability']), reverse=True)
        category_data['class_distribution'] = dict(category_data['class_distribution'].most_common())
        
        category_analysis['categories'][category] = category_data
    
    return category_analysis

def print_category_analysis(analysis, category_name):
    """Print formatted analysis for a specific category"""
    
    if category_name not in analysis['categories']:
        print(f"❌ Category '{category_name}' not found in analysis")
        return
    
    cat_data = analysis['categories'][category_name]
    
    # Category icon mapping
    icons = {
        'weapon': '🔫',
        'accident': '🚗',
        'fight_nano': '👊',
        'fight_small': '🥊',
        'fire_n': '🔥',
        'fire_s': '🔥',
        'nsfw_cls': '🔞',
        'nsfw_seg': '🔞',
        'objects': '📦'
    }
    
    icon = icons.get(category_name, '🔍')
    
    print(f"\n{'='*70}")
    print(f"{icon} {category_name.upper()} DETECTION ANALYSIS")
    print(f"{'='*70}")
    
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Total Detections: {cat_data['total_detections']}")
    print(f"   Total Classifications: {cat_data['total_classifications']}")
    print(f"   Frames with Activity: {cat_data['frames_with_activity']}")
    print(f"   Frames without Activity: {cat_data['frames_without_activity']}")
    print(f"   Activity Rate: {cat_data['activity_rate']:.1f}%")
    
    if cat_data['total_detections'] > 0:
        ds = cat_data['detection_stats']
        print(f"\n🎯 DETECTION STATISTICS:")
        print(f"   Max Confidence: {ds['max_confidence']:.3f}")
        print(f"   Min Confidence: {ds['min_confidence']:.3f}")
        print(f"   Average Confidence: {ds['avg_confidence']:.3f}")
        print(f"   🔴 High Confidence (≥0.7): {ds['high_confidence_count']}")
        print(f"   🟡 Medium Confidence (0.5-0.7): {ds['medium_confidence_count']}")
        print(f"   🟢 Low Confidence (<0.5): {ds['low_confidence_count']}")
    
    if cat_data['total_classifications'] > 0:
        cs = cat_data['classification_stats']
        print(f"\n📋 CLASSIFICATION STATISTICS:")
        print(f"   Max Probability: {cs['max_probability']:.3f}")
        print(f"   Min Probability: {cs['min_probability']:.3f}")
        print(f"   Average Probability: {cs['avg_probability']:.3f}")
        print(f"   🔴 High Probability (≥0.7): {cs['high_probability_count']}")
        print(f"   🟡 Medium Probability (0.5-0.7): {cs['medium_probability_count']}")
        print(f"   🟢 Low Probability (<0.5): {cs['low_probability_count']}")
    
    if cat_data['class_distribution']:
        print(f"\n🏷️  CLASS DISTRIBUTION:")
        for class_name, count in list(cat_data['class_distribution'].items())[:10]:
            print(f"   • {class_name}: {count}")
    
    print(f"\n⏰ TIMELINE - TOP FRAMES:")
    for i, frame in enumerate(cat_data['timeline'][:10], 1):
        max_conf = frame['max_confidence']
        max_prob = frame['max_probability']
        conf_icon = "🔴" if max_conf >= 0.7 else "🟡" if max_conf >= 0.5 else "🟢"
        prob_icon = "🔴" if max_prob >= 0.7 else "🟡" if max_prob >= 0.5 else "🟢"
        
        print(f"   {i}. Second {frame['second']} (Frame {frame['frame_number']})")
        if frame['detection_count'] > 0:
            print(f"      {conf_icon} {frame['detection_count']} detection(s), max conf: {max_conf:.3f}")
        if frame['classification_count'] > 0:
            print(f"      {prob_icon} {frame['classification_count']} classification(s), max prob: {max_prob:.3f}")
    
    print(f"\n🎯 TOP 10 HIGHEST CONFIDENCE DETECTIONS:")
    for i, detection in enumerate(cat_data['all_detections'][:10], 1):
        conf_icon = "🔴" if detection['confidence'] >= 0.7 else "🟡" if detection['confidence'] >= 0.5 else "🟢"
        print(f"   {i}. {conf_icon} Frame {detection['frame_number']} (Second {detection['second']})")
        print(f"      Class: {detection['class_name']}")
        print(f"      Confidence: {detection['confidence']:.3f}")
        print(f"      Position: ({detection['bbox_center'][0]:.0f}, {detection['bbox_center'][1]:.0f})")
    
    if cat_data['all_classifications']:
        print(f"\n🎯 TOP 10 HIGHEST PROBABILITY CLASSIFICATIONS:")
        for i, classification in enumerate(cat_data['all_classifications'][:10], 1):
            prob_icon = "🔴" if classification['probability'] >= 0.7 else "🟡" if classification['probability'] >= 0.5 else "🟢"
            print(f"   {i}. {prob_icon} Frame {classification['frame_number']} (Second {classification['second']})")
            print(f"      Class: {classification['class_name']}")
            print(f"      Probability: {classification['probability']:.3f}")
    
    print(f"{'='*70}")

def print_overall_summary(analysis):
    """Print overall summary across all categories"""
    
    print(f"\n{'='*70}")
    print(f"🎯 OVERALL ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    video_name = Path(analysis['video_info']['file']).name
    print(f"🎥 Video: {video_name}")
    print(f"⏱️  Duration: {analysis['video_info']['duration']:.2f} seconds")
    print(f"🔍 Frames Analyzed: {analysis['video_info']['total_frames_analyzed']}")
    
    # Calculate overall stats
    total_detections = sum(cat['total_detections'] for cat in analysis['categories'].values())
    total_classifications = sum(cat['total_classifications'] for cat in analysis['categories'].values())
    
    print(f"\n📊 GLOBAL STATISTICS:")
    print(f"   Total Detections (all categories): {total_detections}")
    print(f"   Total Classifications (all categories): {total_classifications}")
    
    # Category ranking by activity
    category_scores = []
    unsafe_categories = {'weapon', 'fight_nano', 'fight_small', 'fire_n', 'fire_s', 'nsfw_cls', 'nsfw_seg', 'accident'}
    
    for cat_name, cat_data in analysis['categories'].items():
        score = cat_data['total_detections'] * 2 + cat_data['total_classifications']
        high_conf_dets = cat_data['detection_stats']['high_confidence_count']
        high_prob_cls = cat_data['classification_stats']['high_probability_count']
        
        category_scores.append({
            'name': cat_name,
            'score': score,
            'activity_rate': cat_data['activity_rate'],
            'high_confidence_items': high_conf_dets + high_prob_cls,
            'is_unsafe': cat_name in unsafe_categories and (high_conf_dets > 0 or high_prob_cls > 0)
        })
    
    category_scores.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 CATEGORY RANKING BY ACTIVITY:")
    safety_status = "SAFE"
    
    for i, cat in enumerate(category_scores[:5], 1):
        risk_icon = "🚨" if cat['is_unsafe'] else "✅"
        print(f"   {i}. {risk_icon} {cat['name']}: Score {cat['score']}, Activity {cat['activity_rate']:.1f}%, High-conf items: {cat['high_confidence_items']}")
        
        if cat['is_unsafe']:
            safety_status = "UNSAFE"
    
    print(f"\n🛡️  OVERALL SAFETY ASSESSMENT: {safety_status}")
    print(f"{'='*70}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_category_analysis.py <detailed_results.json> [category1,category2,...]")
        print("Available categories: weapon, accident, fight_nano, fight_small, fire_n, fire_s, nsfw_cls, nsfw_seg, objects")
        print("Use 'all' to analyze all categories")
        sys.exit(1)
    
    detailed_results_path = Path(sys.argv[1])
    categories_to_analyze = sys.argv[2].split(',') if len(sys.argv) > 2 else ['all']
    
    if not detailed_results_path.exists():
        print(f"Error: File not found: {detailed_results_path}")
        sys.exit(1)
    
    print(f"🔍 Analyzing all categories from: {detailed_results_path}")
    
    try:
        analysis = analyze_all_categories(detailed_results_path)
        
        # Print overall summary first
        print_overall_summary(analysis)
        
        # Analyze specific categories or all
        available_categories = ['weapon', 'accident', 'fight_nano', 'fight_small', 'fire_n', 'fire_s', 'nsfw_cls', 'nsfw_seg', 'objects']
        
        if 'all' in categories_to_analyze:
            categories_to_analyze = available_categories
        
        for category in categories_to_analyze:
            if category in available_categories:
                print_category_analysis(analysis, category)
            else:
                print(f"⚠️  Unknown category: {category}")
        
        # Save comprehensive analysis
        input_stem = detailed_results_path.stem.replace('_detailed_results', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = detailed_results_path.parent / f"{input_stem}_comprehensive_analysis_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error analyzing categories: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
