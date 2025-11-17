#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
결과 수집 유틸리티 / Result Collection Utility

벤치마크 출력 파일을 파싱하여 논문용 CSV 형식으로 변환합니다.
Parses benchmark output files and converts them to CSV format for paper.

사용법 / Usage:
    python collect_results.py --input results/raw/ --output results/processed/

작성자 / Author: Research Team
날짜 / Date: 2025-11-17
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm

class BenchmarkResultCollector:
    """
    벤치마크 결과 수집 및 처리 클래스
    Benchmark result collection and processing class
    """

    def __init__(self, input_dir: str, output_dir: str):
        """
        초기화 / Initialize

        Args:
            input_dir: 원본 벤치마크 결과 디렉토리
            output_dir: 처리된 CSV 출력 디렉토리
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 알고리즘 이름 매핑 / Algorithm name mapping
        self.algorithm_names = {
            'seq': 'Duan et al. (순차)',
            'dijkstra': 'Dijkstra',
            'bellman_ford': 'Bellman-Ford',
            'openmp': 'Duan et al. (OpenMP)',
            'mpi': 'Duan et al. (MPI)',
            'cuda': 'Duan et al. (CUDA)',
            'mgap': 'MGAP (제안 기법)'
        }

    def parse_benchmark_file(self, filepath: Path) -> List[Dict]:
        """
        벤치마크 출력 파일 파싱 / Parse benchmark output file

        예상 형식 / Expected format:
            Algorithm: dijkstra
            Dataset: road_network_1M.txt
            Vertices: 1000000
            Edges: 5000000
            Execution Time: 1234.56 ms
            Memory Usage: 512.34 MB
            ...

        Returns:
            파싱된 결과 딕셔너리 리스트
        """
        results = []
        current_result = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 키: 값 형식 파싱
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # 숫자 추출
                    if 'Time' in key:
                        # "1234.56 ms" -> 1234.56
                        match = re.search(r'([\d.]+)', value)
                        if match:
                            current_result[key] = float(match.group(1))
                    elif 'Vertices' in key or 'Edges' in key:
                        match = re.search(r'([\d,]+)', value.replace(',', ''))
                        if match:
                            current_result[key] = int(match.group(1))
                    elif 'Memory' in key:
                        # "512.34 MB" -> 512.34
                        match = re.search(r'([\d.]+)', value)
                        if match:
                            current_result[key] = float(match.group(1))
                    elif 'Speedup' in key:
                        match = re.search(r'([\d.]+)x', value)
                        if match:
                            current_result[key] = float(match.group(1))
                    else:
                        current_result[key] = value

                # 빈 줄이 결과 구분자 / Empty line separates results
                elif line == '' and current_result:
                    results.append(current_result.copy())
                    current_result = {}

            # 마지막 결과 추가
            if current_result:
                results.append(current_result)

        return results

    def create_performance_summary(self, all_results: List[Dict]) -> pd.DataFrame:
        """
        성능 요약 표 생성 / Create performance summary table

        Returns:
            DataFrame with columns: Algorithm, Dataset, Vertices, Edges, Time(ms), Speedup, MTEPS
        """
        data = []

        for result in all_results:
            row = {
                '알고리즘 (Algorithm)': self.algorithm_names.get(
                    result.get('Algorithm', ''),
                    result.get('Algorithm', 'Unknown')
                ),
                '데이터셋 (Dataset)': result.get('Dataset', 'Unknown'),
                '정점 수 (Vertices)': result.get('Vertices', 0),
                '간선 수 (Edges)': result.get('Edges', 0),
                '실행 시간 (Time, ms)': result.get('Execution Time', 0.0),
                '속도 향상 (Speedup)': result.get('Speedup', 1.0),
                '처리량 (MTEPS)': result.get('Throughput (MTEPS)', 0.0)
            }
            data.append(row)

        df = pd.DataFrame(data)

        # MTEPS 계산 (edges / (time_ms * 1000))
        df['처리량 (MTEPS)'] = df['간선 수 (Edges)'] / (df['실행 시간 (Time, ms)'] * 1000)

        return df

    def create_speedup_table(self, all_results: List[Dict]) -> pd.DataFrame:
        """
        속도 향상 표 생성 / Create speedup table

        각 데이터셋별로 알고리즘 간 속도 향상 비교
        Compare speedup across algorithms for each dataset
        """
        # 데이터셋별로 그룹화
        dataset_groups = {}
        for result in all_results:
            dataset = result.get('Dataset', 'Unknown')
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)

        # 각 데이터셋에 대해 속도 향상 계산
        data = []
        for dataset, results in dataset_groups.items():
            # 순차 베이스라인 찾기
            baseline_time = None
            for r in results:
                if r.get('Algorithm') == 'seq' or 'Dijkstra' in r.get('Algorithm', ''):
                    baseline_time = r.get('Execution Time', 1.0)
                    break

            if baseline_time is None:
                # 베이스라인이 없으면 가장 느린 알고리즘 사용
                baseline_time = max(r.get('Execution Time', 1.0) for r in results)

            for r in results:
                algo = self.algorithm_names.get(r.get('Algorithm', ''), r.get('Algorithm', 'Unknown'))
                time_ms = r.get('Execution Time', 1.0)
                speedup = baseline_time / time_ms if time_ms > 0 else 1.0

                data.append({
                    '데이터셋 (Dataset)': dataset,
                    '알고리즘 (Algorithm)': algo,
                    '실행 시간 (ms)': time_ms,
                    '속도 향상 (Speedup)': speedup,
                    '효율 (Efficiency, %)': 0.0  # 나중에 계산
                })

        return pd.DataFrame(data)

    def create_communication_analysis(self, all_results: List[Dict]) -> pd.DataFrame:
        """
        통신 분석 표 생성 / Create communication analysis table

        Multi-GPU 결과에 대한 통신 메트릭
        Communication metrics for Multi-GPU results
        """
        data = []

        for result in all_results:
            algo = result.get('Algorithm', '')

            # Multi-GPU 알고리즘만 포함
            if algo in ['mpi', 'cuda', 'mgap']:
                row = {
                    '알고리즘 (Algorithm)': self.algorithm_names.get(algo, algo),
                    '데이터셋 (Dataset)': result.get('Dataset', 'Unknown'),
                    'GPU 수 (# GPUs)': result.get('GPU Count', 1),
                    '간선 절단 (Edge-Cut)': result.get('Edge-Cut', 0),
                    '통신량 (MB)': result.get('Communication Volume (MB)', 0.0),
                    '통신 시간 (ms)': result.get('Communication Time (ms)', 0.0),
                    '통신 비율 (%)': result.get('Communication Ratio (%)', 0.0),
                    '대역폭 (GB/s)': result.get('Bandwidth (GB/s)', 0.0)
                }
                data.append(row)

        return pd.DataFrame(data)

    def create_scalability_data(self, all_results: List[Dict]) -> pd.DataFrame:
        """
        확장성 데이터 표 생성 / Create scalability data table

        GPU 수에 따른 strong/weak scaling 데이터
        Strong/weak scaling data by GPU count
        """
        data = []

        # GPU 수별로 그룹화
        for result in all_results:
            algo = result.get('Algorithm', '')
            if algo in ['cuda', 'mgap']:
                gpu_count = result.get('GPU Count', 1)
                time_ms = result.get('Execution Time', 0.0)

                data.append({
                    '알고리즘 (Algorithm)': self.algorithm_names.get(algo, algo),
                    '데이터셋 (Dataset)': result.get('Dataset', 'Unknown'),
                    'GPU 수 (# GPUs)': gpu_count,
                    '실행 시간 (ms)': time_ms,
                    '이상적 시간 (ms)': 0.0,  # 나중에 계산
                    '속도 향상 (Speedup)': 0.0,  # 나중에 계산
                    '효율 (Efficiency, %)': 0.0  # 나중에 계산
                })

        df = pd.DataFrame(data)

        # 각 데이터셋별로 1 GPU 베이스라인으로 계산
        for dataset in df['데이터셋 (Dataset)'].unique():
            mask = df['데이터셋 (Dataset)'] == dataset
            baseline_time = df[mask & (df['GPU 수 (# GPUs)'] == 1)]['실행 시간 (ms)'].values

            if len(baseline_time) > 0:
                baseline = baseline_time[0]
                df.loc[mask, '이상적 시간 (ms)'] = baseline / df.loc[mask, 'GPU 수 (# GPUs)']
                df.loc[mask, '속도 향상 (Speedup)'] = baseline / df.loc[mask, '실행 시간 (ms)']
                df.loc[mask, '효율 (Efficiency, %)'] = (
                    df.loc[mask, '속도 향상 (Speedup)'] / df.loc[mask, 'GPU 수 (# GPUs)'] * 100
                )

        return df

    def create_memory_usage_table(self, all_results: List[Dict]) -> pd.DataFrame:
        """
        메모리 사용량 표 생성 / Create memory usage table
        """
        data = []

        for result in all_results:
            row = {
                '알고리즘 (Algorithm)': self.algorithm_names.get(
                    result.get('Algorithm', ''),
                    result.get('Algorithm', 'Unknown')
                ),
                '데이터셋 (Dataset)': result.get('Dataset', 'Unknown'),
                '정점 수 (Vertices)': result.get('Vertices', 0),
                '간선 수 (Edges)': result.get('Edges', 0),
                'CPU 메모리 (MB)': result.get('Memory Usage (MB)', 0.0),
                'GPU 메모리 (MB)': result.get('GPU Memory (MB)', 0.0),
                '총 메모리 (MB)': result.get('Total Memory (MB)', 0.0),
                '메모리 효율 (%)': 0.0  # 이론값 대비
            }

            # 이론적 메모리 사용량: O(n + m) * sizeof(data)
            # 정점: 8 bytes (거리) + 4 bytes (predecessor)
            # 간선: 4 bytes (target) + 8 bytes (weight)
            theoretical = (row['정점 수 (Vertices)'] * 12 + row['간선 수 (Edges)'] * 12) / (1024 * 1024)
            if row['총 메모리 (MB)'] > 0:
                row['메모리 효율 (%)'] = (theoretical / row['총 메모리 (MB)']) * 100

            data.append(row)

        return pd.DataFrame(data)

    def collect_all_results(self) -> Dict[str, pd.DataFrame]:
        """
        모든 결과 수집 및 표 생성 / Collect all results and create tables

        Returns:
            각 표의 이름을 키로 하는 DataFrame 딕셔너리
        """
        print("📊 벤치마크 결과 수집 중... / Collecting benchmark results...")

        # 모든 결과 파일 찾기
        result_files = list(self.input_dir.glob('**/*.txt')) + \
                      list(self.input_dir.glob('**/*.log')) + \
                      list(self.input_dir.glob('**/*.json'))

        all_results = []

        for filepath in tqdm(result_files, desc="파싱 중 / Parsing"):
            if filepath.suffix == '.json':
                # JSON 파일 직접 로드
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_results.extend(data)
                    else:
                        all_results.append(data)
            else:
                # 텍스트 파일 파싱
                parsed = self.parse_benchmark_file(filepath)
                all_results.extend(parsed)

        print(f"✅ {len(all_results)}개 결과 수집 완료 / Collected {len(all_results)} results")

        # 각 표 생성
        tables = {}

        print("📈 성능 요약 표 생성 중... / Creating performance summary...")
        tables['performance_summary'] = self.create_performance_summary(all_results)

        print("⚡ 속도 향상 표 생성 중... / Creating speedup table...")
        tables['speedup_table'] = self.create_speedup_table(all_results)

        print("📡 통신 분석 표 생성 중... / Creating communication analysis...")
        tables['communication_analysis'] = self.create_communication_analysis(all_results)

        print("📊 확장성 데이터 생성 중... / Creating scalability data...")
        tables['scalability_data'] = self.create_scalability_data(all_results)

        print("💾 메모리 사용량 표 생성 중... / Creating memory usage table...")
        tables['memory_usage'] = self.create_memory_usage_table(all_results)

        return tables

    def save_tables(self, tables: Dict[str, pd.DataFrame]):
        """
        모든 표를 CSV로 저장 / Save all tables to CSV
        """
        print("\n💾 CSV 파일 저장 중... / Saving CSV files...")

        for name, df in tables.items():
            output_path = self.output_dir / f"{name}.csv"
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✅ {output_path}")

        # 요약 통계도 저장
        summary_path = self.output_dir / "summary_statistics.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("벤치마크 결과 요약 통계 / Benchmark Result Summary Statistics\n")
            f.write("=" * 80 + "\n\n")

            for name, df in tables.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{name.upper()}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"\n행 개수 / Row count: {len(df)}\n")
                f.write(f"열 개수 / Column count: {len(df.columns)}\n\n")
                f.write("기술 통계 / Descriptive statistics:\n")
                f.write(str(df.describe()) + "\n")

        print(f"  ✅ {summary_path}")
        print("\n✅ 모든 결과 저장 완료! / All results saved successfully!")

def main():
    """메인 함수 / Main function"""
    parser = argparse.ArgumentParser(
        description='벤치마크 결과 수집 및 CSV 변환 / Collect benchmark results and convert to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제 / Examples:
  python collect_results.py --input results/raw --output results/processed
  python collect_results.py -i ../results -o ../paper_results
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='results/raw',
        help='원본 벤치마크 결과 디렉토리 / Input directory with raw benchmark results'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/processed',
        help='처리된 CSV 출력 디렉토리 / Output directory for processed CSV files'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("📊 SSSP 벤치마크 결과 수집 유틸리티")
    print("   SSSP Benchmark Result Collection Utility")
    print("=" * 80)
    print(f"\n입력 디렉토리 / Input directory: {args.input}")
    print(f"출력 디렉토리 / Output directory: {args.output}\n")

    # 결과 수집기 생성
    collector = BenchmarkResultCollector(args.input, args.output)

    # 모든 결과 수집 및 표 생성
    tables = collector.collect_all_results()

    # CSV로 저장
    collector.save_tables(tables)

    print("\n" + "=" * 80)
    print("🎉 완료! / Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
