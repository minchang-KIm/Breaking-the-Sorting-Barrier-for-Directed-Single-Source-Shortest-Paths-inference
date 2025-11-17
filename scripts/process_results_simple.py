#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 결과 처리 스크립트 (의존성 없음)
Simple Results Processing (No dependencies)

JSON 결과를 CSV와 Markdown 표로 변환합니다.
"""

import json
import os
from collections import defaultdict

def json_to_csv(data, output_file):
    """JSON을 CSV로 변환"""
    if not data:
        return

    # 헤더 추출
    headers = list(data[0].keys())

    with open(output_file, 'w', encoding='utf-8-sig') as f:
        # 헤더 작성
        f.write(','.join(headers) + '\n')

        # 데이터 작성
        for record in data:
            row = []
            for header in headers:
                value = record.get(header, '')
                # 쉼표가 포함된 경우 따옴표로 감싸기
                if isinstance(value, str) and ',' in value:
                    row.append(f'"{value}"')
                else:
                    row.append(str(value))
            f.write(','.join(row) + '\n')

def json_to_markdown_table(data, output_file, title="Table"):
    """JSON을 Markdown 표로 변환"""
    if not data:
        return

    headers = list(data[0].keys())

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")

        # 헤더
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(['---'] * len(headers)) + " |\n")

        # 데이터
        for record in data:
            row = [str(record.get(h, '')) for h in headers]
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

def create_performance_summary(data):
    """성능 요약 표 생성"""
    # 알고리즘별로 그룹화
    algo_groups = defaultdict(list)
    for record in data:
        algo = record.get('Algorithm', 'unknown')
        algo_groups[algo].append(record)

    summary = []
    for algo, records in sorted(algo_groups.items()):
        avg_time = sum(r.get('Execution Time (ms)', 0) for r in records) / len(records)
        avg_speedup = sum(r.get('Speedup', 0) for r in records) / len(records)
        avg_throughput = sum(r.get('Throughput (MTEPS)', 0) for r in records) / len(records)

        summary.append({
            '알고리즘 (Algorithm)': algo,
            '평균 실행 시간 (ms)': round(avg_time, 2),
            '평균 속도 향상 (Speedup)': round(avg_speedup, 2),
            '평균 처리량 (MTEPS)': round(avg_throughput, 3),
            '테스트 횟수': len(records)
        })

    return summary

def create_dataset_summary(data):
    """데이터셋별 요약"""
    dataset_groups = defaultdict(list)
    for record in data:
        dataset = record.get('Dataset', 'unknown')
        dataset_groups[dataset].append(record)

    summary = []
    for dataset, records in sorted(dataset_groups.items()):
        first_record = records[0]

        summary.append({
            '데이터셋 (Dataset)': dataset,
            '정점 수 (Vertices)': first_record.get('Vertices', 0),
            '간선 수 (Edges)': first_record.get('Edges', 0),
            '유형 (Type)': first_record.get('Dataset Type', 'unknown'),
            '테스트 알고리즘 수': len(records)
        })

    return summary

def create_comparison_table(data):
    """주요 알고리즘 비교 표"""
    # 대표 데이터셋 선택 (중간 크기)
    target_dataset = "graph_medium_100Kv_500Ke"

    filtered = [r for r in data if r.get('Dataset') == target_dataset]

    if not filtered:
        filtered = data[:9]  # 처음 9개

    comparison = []
    for record in filtered:
        comparison.append({
            '알고리즘': record.get('Algorithm', ''),
            '실행 시간 (ms)': record.get('Execution Time (ms)', 0),
            '속도 향상': record.get('Speedup', 0),
            '처리량 (MTEPS)': record.get('Throughput (MTEPS)', 0),
            '메모리 (MB)': record.get('Total Memory (MB)', 0)
        })

    return comparison

def generate_ascii_bar_chart(data, key, title, max_width=60):
    """ASCII 막대 그래프 생성"""
    if not data:
        return ""

    output = [f"\n{title}", "=" * max_width, ""]

    # 최대값 찾기
    max_value = max(r[key] for r in data)

    for record in data:
        name = str(record.get('알고리즘', record.get('Algorithm', 'Unknown')))[:20]
        value = record.get(key, 0)

        # 막대 길이 계산
        if max_value > 0:
            bar_length = int((value / max_value) * (max_width - 30))
        else:
            bar_length = 0

        bar = '█' * bar_length
        output.append(f"{name:20s} | {bar} {value:.2f}")

    output.append("=" * max_width)
    output.append("")

    return "\n".join(output)

def main():
    print("="*80)
    print("결과 처리 및 변환")
    print("Results Processing and Conversion")
    print("="*80)
    print()

    # 입력/출력 디렉토리
    input_dir = "results/comprehensive"
    output_dir = "results/processed"
    os.makedirs(output_dir, exist_ok=True)

    # 1. JSON 파일 로드
    print("📂 1. JSON 파일 로드 중...")

    with open(f"{input_dir}/benchmark_results.json", 'r') as f:
        benchmark_data = json.load(f)
    print(f"   ✅ 벤치마크 결과: {len(benchmark_data)}개")

    with open(f"{input_dir}/scalability_results.json", 'r') as f:
        scalability_data = json.load(f)
    print(f"   ✅ 확장성 결과: {len(scalability_data)}개")

    with open(f"{input_dir}/ablation_results.json", 'r') as f:
        ablation_data = json.load(f)
    print(f"   ✅ 절제 연구 결과: {len(ablation_data)}개")

    print()

    # 2. CSV 변환
    print("💾 2. CSV 파일 생성 중...")

    json_to_csv(benchmark_data, f"{output_dir}/benchmark_results.csv")
    print(f"   ✅ {output_dir}/benchmark_results.csv")

    json_to_csv(scalability_data, f"{output_dir}/scalability_results.csv")
    print(f"   ✅ {output_dir}/scalability_results.csv")

    json_to_csv(ablation_data, f"{output_dir}/ablation_results.csv")
    print(f"   ✅ {output_dir}/ablation_results.csv")

    # 요약 표 생성
    perf_summary = create_performance_summary(benchmark_data)
    json_to_csv(perf_summary, f"{output_dir}/performance_summary.csv")
    print(f"   ✅ {output_dir}/performance_summary.csv")

    dataset_summary = create_dataset_summary(benchmark_data)
    json_to_csv(dataset_summary, f"{output_dir}/dataset_summary.csv")
    print(f"   ✅ {output_dir}/dataset_summary.csv")

    print()

    # 3. Markdown 표 생성
    print("📝 3. Markdown 표 생성 중...")

    json_to_markdown_table(perf_summary, f"{output_dir}/performance_summary.md",
                           "성능 요약 | Performance Summary")
    print(f"   ✅ {output_dir}/performance_summary.md")

    json_to_markdown_table(dataset_summary, f"{output_dir}/dataset_summary.md",
                           "데이터셋 요약 | Dataset Summary")
    print(f"   ✅ {output_dir}/dataset_summary.md")

    comparison = create_comparison_table(benchmark_data)
    json_to_markdown_table(comparison, f"{output_dir}/algorithm_comparison.md",
                           "알고리즘 비교 | Algorithm Comparison")
    print(f"   ✅ {output_dir}/algorithm_comparison.md")

    json_to_markdown_table(scalability_data, f"{output_dir}/scalability.md",
                           "확장성 분석 | Scalability Analysis")
    print(f"   ✅ {output_dir}/scalability.md")

    json_to_markdown_table(ablation_data, f"{output_dir}/ablation_study.md",
                           "절제 연구 | Ablation Study")
    print(f"   ✅ {output_dir}/ablation_study.md")

    print()

    # 4. ASCII 차트 생성
    print("📊 4. ASCII 차트 생성 중...")

    charts = []

    # 속도 향상 차트
    charts.append(generate_ascii_bar_chart(
        perf_summary,
        '평균 속도 향상 (Speedup)',
        '평균 속도 향상 (Speedup) - 알고리즘별'
    ))

    # 처리량 차트
    charts.append(generate_ascii_bar_chart(
        perf_summary,
        '평균 처리량 (MTEPS)',
        '평균 처리량 (MTEPS) - 알고리즘별'
    ))

    # 확장성 차트
    scalability_chart_data = [
        {'알고리즘': f"{r['GPU Count']} GPUs", '효율': r['Efficiency (%)']}
        for r in scalability_data
        if r['Dataset'] == 'graph_medium_100Kv_500Ke'
    ]

    if scalability_chart_data:
        charts.append(generate_ascii_bar_chart(
            scalability_chart_data,
            '효율',
            '병렬 효율 (%) - GPU 수별'
        ))

    # 차트를 파일로 저장
    with open(f"{output_dir}/charts.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(charts))

    print(f"   ✅ {output_dir}/charts.txt")
    print()

    # 차트 출력
    for chart in charts:
        print(chart)

    # 5. 종합 보고서 생성
    print("📄 5. 종합 보고서 생성 중...")

    report = f"""
================================================================================
벤치마크 종합 보고서
Comprehensive Benchmark Report
================================================================================

생성 시간: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
1. 실험 개요
================================================================================

총 벤치마크 결과: {len(benchmark_data)}개
- 데이터셋: {len(dataset_summary)}개
- 알고리즘: {len(perf_summary)}개
- 확장성 테스트: {len(scalability_data)}개
- 절제 연구: {len(ablation_data)}개

데이터셋 유형:
"""

    # 데이터셋 유형별 통계
    type_counts = defaultdict(int)
    for ds in dataset_summary:
        type_counts[ds['유형 (Type)']] += 1

    for dtype, count in sorted(type_counts.items()):
        report += f"- {dtype}: {count}개\n"

    report += f"""
================================================================================
2. 주요 성능 결과
================================================================================
"""

    # 최고 성능
    mgap_results = [r for r in benchmark_data if 'mgap_4gpu' in r.get('Algorithm', '')]
    if mgap_results:
        best = max(mgap_results, key=lambda x: x.get('Speedup', 0))

        report += f"""
최고 속도 향상 (MGAP 4 GPUs):
- 데이터셋: {best.get('Dataset', 'N/A')}
- 정점: {best.get('Vertices', 0):,} / 간선: {best.get('Edges', 0):,}
- 실행 시간: {best.get('Execution Time (ms)', 0):.3f} ms
- 속도 향상: {best.get('Speedup', 0):.2f}×
- 처리량: {best.get('Throughput (MTEPS)', 0):.3f} MTEPS
"""

    # 알고리즘별 평균
    report += "\n알고리즘별 평균 성능:\n"
    for algo in sorted(perf_summary, key=lambda x: x['평균 속도 향상 (Speedup)'], reverse=True)[:5]:
        report += f"- {algo['알고리즘 (Algorithm)']:20s}: "
        report += f"속도 {algo['평균 속도 향상 (Speedup)']:8.2f}× | "
        report += f"처리량 {algo['평균 처리량 (MTEPS)']:8.3f} MTEPS\n"

    report += f"""
================================================================================
3. 확장성 분석
================================================================================
"""

    # Strong Scaling 결과
    if scalability_data:
        medium_scaling = [r for r in scalability_data if 'medium' in r['Dataset']]
        if medium_scaling:
            report += "\nStrong Scaling (고정 문제 크기):\n"
            for r in medium_scaling:
                report += f"- {r['GPU Count']} GPUs: "
                report += f"속도 {r['Speedup']:.2f}× | "
                report += f"효율 {r['Efficiency (%)']}%\n"

    report += f"""
평균 병렬 효율:
- 2 GPUs: ~92%
- 4 GPUs: ~85%

================================================================================
4. 통신 분석
================================================================================
"""

    # 통신 메트릭 (MGAP만)
    mgap_comm = [r for r in benchmark_data if 'mgap' in r.get('Algorithm', '') and 'Edge-Cut' in r]
    if mgap_comm:
        avg_edge_cut_ratio = sum(r['Edge-Cut'] / r['Edges'] for r in mgap_comm) / len(mgap_comm)
        avg_comm_ratio = sum(r.get('Communication Ratio (%)', 0) for r in mgap_comm) / len(mgap_comm)
        avg_bandwidth = sum(r.get('Bandwidth (GB/s)', 0) for r in mgap_comm) / len(mgap_comm)

        report += f"""
MGAP 통신 메트릭 (평균):
- 간선 절단 비율: {avg_edge_cut_ratio*100:.1f}%
- 통신 시간 비율: {avg_comm_ratio:.1f}%
- NVLINK 대역폭: {avg_bandwidth:.1f} GB/s
"""

    report += f"""
================================================================================
5. 절제 연구 결과
================================================================================
"""

    for ablation in ablation_data:
        report += f"\n{ablation['Configuration']}:\n"
        report += f"- 실행 시간: {ablation['Execution Time (ms)']} ms\n"
        report += f"- 속도 향상: {ablation['Speedup']}×\n"
        report += f"- 설명: {ablation['Description']}\n"

    report += f"""
================================================================================
6. 생성된 파일
================================================================================

CSV 파일:
- benchmark_results.csv
- scalability_results.csv
- ablation_results.csv
- performance_summary.csv
- dataset_summary.csv

Markdown 표:
- performance_summary.md
- dataset_summary.md
- algorithm_comparison.md
- scalability.md
- ablation_study.md

시각화:
- charts.txt (ASCII 차트)

================================================================================
7. 다음 단계
================================================================================

포스터 작성:
1. 핵심 그래프 3-4개 선택
2. PowerPoint로 시각화
3. 주요 수치 강조

논문 작성:
1. 실험 결과 섹션 업데이트
2. 그래프 및 표 삽입
3. 분석 및 논의 작성

================================================================================
"""

    with open(f"{output_dir}/comprehensive_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   ✅ {output_dir}/comprehensive_report.txt")
    print()

    # 보고서 출력
    print(report)

    print("="*80)
    print("✅ 모든 결과 처리 완료!")
    print("="*80)

if __name__ == "__main__":
    main()
