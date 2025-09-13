#!/usr/bin/env python
"""
Create filtered results from detailed video analysis
Extracts top 3 categories, high-confidence objects, and detections
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

def analyze_and_filter_results(detailed_results_path, min_confidence=0.7, min_probability=0.7):
    """
    Analyze detailed results and create filtered summary
    """
    
    # Load detailed results
    with open(detailed_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize analysis structures
    model_stats = defaultdict(lambda: {
        'total_detections': 0,
        'high_conf_detections': 0,
        'max_confidence': 0.0,
        'avg_confidence': 0.0,
        'classes': Counter(),
        'high_conf_objects': [],
        'frames_detected': set()
    })
    
    classification_stats = defaultdict(lambda: {
        'total_classifications': 0,
        'high_prob_classifications': 0,
        'max_probability': 0.0,
        'avg_probability': 0.0,
        'classes': Counter(),
        'high_prob_classes': [],
        'frames_detected': set()
    })
    
    # Process each frame
    for frame_data in data['frame_results']:
        frame_num = frame_data['frame_number']
        second = frame_data['second']
        
        for model_name, model_result in frame_data['model_results'].items():
            if not model_result['detected']:
                continue
                
            # Process detections (bounding boxes)
            for detection in model_result['detections']:
                conf = detection['confidence']
                class_name = detection['class_name']
                
                model_stats[model_name]['total_detections'] += 1
                model_stats[model_name]['max_confidence'] = max(model_stats[model_name]['max_confidence'], conf)
                model_stats[model_name]['classes'][class_name] += 1
                model_stats[model_name]['frames_detected'].add(frame_num)
                
                if conf >= min_confidence:
                    model_stats[model_name]['high_conf_detections'] += 1
                    model_stats[model_name]['high_conf_objects'].append({
                        'class': class_name,
                        'confidence': conf,
                        'frame': frame_num,
                        'second': second,
                        'bbox': detection['bbox']
                    })
            
            # Process classifications
            for classification in model_result['classifications']:
                prob = classification['probability']
                class_name = classification['class_name']
                
                classification_stats[model_name]['total_classifications'] += 1
                classification_stats[model_name]['max_probability'] = max(classification_stats[model_name]['max_probability'], prob)
                classification_stats[model_name]['classes'][class_name] += 1
                classification_stats[model_name]['frames_detected'].add(frame_num)
                
                if prob >= min_probability:
                    classification_stats[model_name]['high_prob_classifications'] += 1
                    classification_stats[model_name]['high_prob_classes'].append({
                        'class': class_name,
                        'probability': prob,
                        'frame': frame_num,
                        'second': second
                    })
    
    # Calculate averages and convert sets to counts
    for model_name, stats in model_stats.items():
        if stats['total_detections'] > 0:
            total_conf = sum(obj['confidence'] for obj in stats['high_conf_objects'])
            stats['avg_confidence'] = total_conf / len(stats['high_conf_objects']) if stats['high_conf_objects'] else 0
        stats['frames_detected'] = len(stats['frames_detected'])
        stats['classes'] = dict(stats['classes'].most_common())
    
    for model_name, stats in classification_stats.items():
        if stats['total_classifications'] > 0:
            total_prob = sum(cls['probability'] for cls in stats['high_prob_classes'])
            stats['avg_probability'] = total_prob / len(stats['high_prob_classes']) if stats['high_prob_classes'] else 0
        stats['frames_detected'] = len(stats['frames_detected'])
        stats['classes'] = dict(stats['classes'].most_common())
    
    # Determine top 3 categories by total activity (detections + classifications)
    category_scores = {}
    for model_name in data['models_used']:
        score = 0
        
        # Detection score
        if model_name in model_stats:
            score += model_stats[model_name]['high_conf_detections'] * 2  # Weight detections higher
            score += model_stats[model_name]['frames_detected']
        
        # Classification score  
        if model_name in classification_stats:
            score += classification_stats[model_name]['high_prob_classifications']
            score += classification_stats[model_name]['frames_detected']
        
        category_scores[model_name] = score
    
    # Get top 3 categories
    top_3_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Create filtered results
    filtered_results = {
        'video_info': {
            'file': data['video_path'],
            'duration': data['video_info']['duration'],
            'total_frames': data['video_info']['total_frames'],
            'frames_analyzed': data['total_frames_processed']
        },
        'analysis_metadata': {
            'processing_time': data['processing_time'],
            'confidence_threshold': data['confidence_threshold'],
            'filter_confidence': min_confidence,
            'filter_probability': min_probability,
            'timestamp': datetime.now().isoformat()
        },
        'top_3_categories': [],
        'high_confidence_objects': [],
        'high_probability_classifications': [],
        'safety_assessment': 'SAFE',
        'summary_stats': {
            'total_high_conf_detections': 0,
            'total_high_prob_classifications': 0,
            'frames_with_detections': set(),
            'most_detected_objects': {},
            'most_classified_content': {}
        }
    }
    
    # Process top 3 categories
    unsafe_categories = {'weapon', 'fight_nano', 'fight_small', 'nsfw_cls', 'nsfw_seg', 'fire_n', 'fire_s', 'accident'}
    
    for i, (category, score) in enumerate(top_3_categories, 1):
        category_data = {
            'rank': i,
            'category': category,
            'activity_score': score,
            'detection_stats': {},
            'classification_stats': {},
            'safety_risk': category in unsafe_categories
        }
        
        # Add detection stats if available
        if category in model_stats:
            stats = model_stats[category]
            category_data['detection_stats'] = {
                'total_detections': stats['total_detections'],
                'high_confidence_detections': stats['high_conf_detections'],
                'max_confidence': stats['max_confidence'],
                'avg_confidence': stats['avg_confidence'],
                'frames_detected': stats['frames_detected'],
                'top_classes': dict(list(stats['classes'].items())[:5])
            }
            
            # Add high-confidence objects to global list
            for obj in stats['high_conf_objects']:
                obj['model'] = category
                filtered_results['high_confidence_objects'].append(obj)
                filtered_results['summary_stats']['frames_with_detections'].add(obj['frame'])
        
        # Add classification stats if available
        if category in classification_stats:
            stats = classification_stats[category]
            category_data['classification_stats'] = {
                'total_classifications': stats['total_classifications'],
                'high_probability_classifications': stats['high_prob_classifications'],
                'max_probability': stats['max_probability'],
                'avg_probability': stats['avg_probability'],
                'frames_detected': stats['frames_detected'],
                'top_classes': dict(list(stats['classes'].items())[:5])
            }
            
            # Add high-probability classifications to global list
            for cls in stats['high_prob_classes']:
                cls['model'] = category
                filtered_results['high_probability_classifications'].append(cls)
                filtered_results['summary_stats']['frames_with_detections'].add(cls['frame'])
        
        # Check if unsafe
        if category_data['safety_risk'] and (
            category_data['detection_stats'].get('high_confidence_detections', 0) > 0 or
            category_data['classification_stats'].get('high_probability_classifications', 0) > 0
        ):
            filtered_results['safety_assessment'] = 'UNSAFE'
        
        filtered_results['top_3_categories'].append(category_data)
    
    # Sort objects and classifications by confidence/probability
    filtered_results['high_confidence_objects'].sort(key=lambda x: x['confidence'], reverse=True)
    filtered_results['high_probability_classifications'].sort(key=lambda x: x['probability'], reverse=True)
    
    # Calculate summary stats
    filtered_results['summary_stats']['total_high_conf_detections'] = len(filtered_results['high_confidence_objects'])
    filtered_results['summary_stats']['total_high_prob_classifications'] = len(filtered_results['high_probability_classifications'])
    filtered_results['summary_stats']['frames_with_detections'] = len(filtered_results['summary_stats']['frames_with_detections'])
    
    # Most detected objects and classifications
    all_objects = Counter()
    all_classifications = Counter()
    
    for obj in filtered_results['high_confidence_objects']:
        all_objects[obj['class']] += 1
    
    for cls in filtered_results['high_probability_classifications']:
        all_classifications[cls['class']] += 1
    
    filtered_results['summary_stats']['most_detected_objects'] = dict(all_objects.most_common(10))
    filtered_results['summary_stats']['most_classified_content'] = dict(all_classifications.most_common(10))
    
    return filtered_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_filtered_results.py <detailed_results.json> [min_confidence] [min_probability]")
        sys.exit(1)
    
    detailed_results_path = Path(sys.argv[1])
    min_confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    min_probability = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    
    if not detailed_results_path.exists():
        print(f"Error: File not found: {detailed_results_path}")
        sys.exit(1)
    
    print(f"🔍 Analyzing results from: {detailed_results_path}")
    print(f"📊 Filter thresholds - Confidence: {min_confidence}, Probability: {min_probability}")
    
    try:
        filtered_results = analyze_and_filter_results(detailed_results_path, min_confidence, min_probability)
        
        # Generate output filename
        input_stem = detailed_results_path.stem.replace('_detailed_results', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = detailed_results_path.parent / f"{input_stem}_filtered_final_{timestamp}.json"
        
        # Save filtered results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Filtered results saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 FILTERED ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"🎥 Video: {Path(filtered_results['video_info']['file']).name}")
        print(f"⏱️  Duration: {filtered_results['video_info']['duration']:.2f}s")
        print(f"🔍 Frames Analyzed: {filtered_results['video_info']['frames_analyzed']}")
        print(f"🛡️  Safety Assessment: {filtered_results['safety_assessment']}")
        
        print(f"\n🏆 TOP 3 CATEGORIES:")
        for cat in filtered_results['top_3_categories']:
            risk_icon = "🚨" if cat['safety_risk'] else "✅"
            print(f"  {cat['rank']}. {risk_icon} {cat['category']} (Score: {cat['activity_score']})")
            
            if cat['detection_stats']:
                stats = cat['detection_stats']
                print(f"     🎯 Detections: {stats['high_confidence_detections']}/{stats['total_detections']} high-conf")
                print(f"     📈 Max confidence: {stats['max_confidence']:.3f}")
                if stats['top_classes']:
                    top_class = list(stats['top_classes'].items())[0]
                    print(f"     🔝 Top class: {top_class[0]} ({top_class[1]} detections)")
            
            if cat['classification_stats']:
                stats = cat['classification_stats']
                print(f"     🎯 Classifications: {stats['high_probability_classifications']}/{stats['total_classifications']} high-prob")
                print(f"     📈 Max probability: {stats['max_probability']:.3f}")
                if stats['top_classes']:
                    top_class = list(stats['top_classes'].items())[0]
                    print(f"     🔝 Top class: {top_class[0]} ({top_class[1]} classifications)")
        
        print(f"\n📋 HIGH-CONFIDENCE DETECTIONS: {filtered_results['summary_stats']['total_high_conf_detections']}")
        for i, obj in enumerate(filtered_results['high_confidence_objects'][:5], 1):
            print(f"  {i}. {obj['model']}: {obj['class']} ({obj['confidence']:.3f}) - Frame {obj['frame']}")
        if len(filtered_results['high_confidence_objects']) > 5:
            print(f"  ... and {len(filtered_results['high_confidence_objects']) - 5} more")
        
        print(f"\n📋 HIGH-PROBABILITY CLASSIFICATIONS: {filtered_results['summary_stats']['total_high_prob_classifications']}")
        for i, cls in enumerate(filtered_results['high_probability_classifications'][:5], 1):
            print(f"  {i}. {cls['model']}: {cls['class']} ({cls['probability']:.3f}) - Frame {cls['frame']}")
        if len(filtered_results['high_probability_classifications']) > 5:
            print(f"  ... and {len(filtered_results['high_probability_classifications']) - 5} more")
        
        print(f"\n📊 MOST DETECTED OBJECTS:")
        for obj, count in list(filtered_results['summary_stats']['most_detected_objects'].items())[:5]:
            print(f"  • {obj}: {count} detections")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
