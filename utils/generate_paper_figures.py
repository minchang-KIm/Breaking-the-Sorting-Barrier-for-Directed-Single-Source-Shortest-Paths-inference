#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
논문 그래프 생성 유틸리티 / Paper Figure Generation Utility

CSV 데이터를 읽어 논문용 고품질 그래프(PDF + PNG)를 생성합니다.
Reads CSV data and generates high-quality figures (PDF + PNG) for paper.

사용법 / Usage:
    python generate_paper_figures.py --data paper_results/ --output figures/

작성자 / Author: Research Team
날짜 / Date: 2025-11-17
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# 한국어 폰트 설정 / Korean font setup
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 논문 품질 설정 / Paper quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# 색상 팔레트 / Color palette
COLORS = {
    'Dijkstra': '#1f77b4',
    'Bellman-Ford': '#ff7f0e',
    'Duan et al. (순차)': '#2ca02c',
    'Duan et al. (OpenMP)': '#d62728',
    'Duan et al. (MPI)': '#9467bd',
    'Duan et al. (CUDA)': '#8c564b',
    'MGAP (제안 기법)': '#e377c2'
}

class PaperFigureGenerator:
    """
    논문 그래프 생성 클래스
    Paper figure generation class
    """

    def __init__(self, data_dir: str, output_dir: str, language: str = 'korean', dpi: int = 300):
        """
        초기화 / Initialize

        Args:
            data_dir: CSV 데이터 디렉토리
            output_dir: 그래프 출력 디렉토리
            language: 'korean' or 'english'
            dpi: 해상도 (기본 300)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        self.dpi = dpi

        # 출력 디렉토리 생성
        (self.output_dir / 'pdf').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'png').mkdir(parents=True, exist_ok=True)

        # 라벨 설정
        self.labels = self._get_labels()

    def _get_labels(self) -> Dict[str, str]:
        """언어별 라벨 반환 / Return labels by language"""
        if self.language == 'korean':
            return {
                'algorithm': '알고리즘',
                'dataset': '데이터셋',
                'time': '실행 시간 (ms)',
                'speedup': '속도 향상',
                'throughput': '처리량 (MTEPS)',
                'memory': '메모리 (MB)',
                'edge_cut': '간선 절단',
                'communication': '통신량 (MB)',
                'efficiency': '효율 (%)',
                'gpu_count': 'GPU 수',
                'bandwidth': '대역폭 (GB/s)'
            }
        else:
            return {
                'algorithm': 'Algorithm',
                'dataset': 'Dataset',
                'time': 'Execution Time (ms)',
                'speedup': 'Speedup',
                'throughput': 'Throughput (MTEPS)',
                'memory': 'Memory (MB)',
                'edge_cut': 'Edge-Cut',
                'communication': 'Communication Volume (MB)',
                'efficiency': 'Efficiency (%)',
                'gpu_count': '# GPUs',
                'bandwidth': 'Bandwidth (GB/s)'
            }

    def save_figure(self, fig: plt.Figure, name: str):
        """
        그래프를 PDF와 PNG로 저장 / Save figure as PDF and PNG
        """
        pdf_path = self.output_dir / 'pdf' / f'{name}.pdf'
        png_path = self.output_dir / 'png' / f'{name}.png'

        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=self.dpi)
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=self.dpi)

        print(f"  ✅ {pdf_path.name}")
        plt.close(fig)

    def figure1_execution_time_comparison(self, df: pd.DataFrame):
        """
        그림 1: 실행 시간 비교 (막대 그래프)
        Figure 1: Execution time comparison (bar chart)
        """
        print("📊 그림 1: 실행 시간 비교 생성 중...")

        # 중간 크기 데이터셋만 사용
        df_filtered = df[df['정점 수 (Vertices)'].between(100000, 2000000)]

        fig, ax = plt.subplots(figsize=(10, 6))

        # 알고리즘별로 그룹화
        algorithms = df_filtered['알고리즘 (Algorithm)'].unique()
        datasets = df_filtered['데이터셋 (Dataset)'].unique()[:5]  # 최대 5개

        x = np.arange(len(datasets))
        width = 0.12
        multiplier = 0

        for algorithm in algorithms:
            data = df_filtered[df_filtered['알고리즘 (Algorithm)'] == algorithm]
            times = [data[data['데이터셋 (Dataset)'] == d]['실행 시간 (Time, ms)'].values[0]
                    if len(data[data['데이터셋 (Dataset)'] == d]) > 0 else 0
                    for d in datasets]

            offset = width * multiplier
            color = COLORS.get(algorithm, f'C{multiplier}')
            ax.bar(x + offset, times, width, label=algorithm, color=color, alpha=0.8)
            multiplier += 1

        ax.set_xlabel(self.labels['dataset'], fontweight='bold')
        ax.set_ylabel(self.labels['time'], fontweight='bold')
        ax.set_title('실행 시간 비교 / Execution Time Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels([d.replace('.txt', '').replace('_', ' ')[:15] for d in datasets],
                          rotation=45, ha='right')
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

        self.save_figure(fig, '1_execution_time_comparison')

    def figure2_speedup_vs_gpus(self, df: pd.DataFrame):
        """
        그림 2: GPU 수에 따른 속도 향상 (꺾은선 그래프)
        Figure 2: Speedup vs GPU count (line plot)
        """
        print("📊 그림 2: 속도 향상 그래프 생성 중...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 데이터셋별로 선 그리기
        for dataset in df['데이터셋 (Dataset)'].unique():
            data = df[df['데이터셋 (Dataset)'] == dataset].sort_values('GPU 수 (# GPUs)')
            if len(data) > 1:
                ax.plot(data['GPU 수 (# GPUs)'], data['속도 향상 (Speedup)'],
                       marker='o', linewidth=2, markersize=8,
                       label=dataset.replace('.txt', '').replace('_', ' ')[:20])

        # 이상적인 선형 속도 향상 (참조선)
        max_gpus = df['GPU 수 (# GPUs)'].max()
        ax.plot([1, max_gpus], [1, max_gpus], 'k--', linewidth=1.5,
               label='이상적 선형 / Ideal Linear', alpha=0.5)

        ax.set_xlabel(self.labels['gpu_count'], fontweight='bold')
        ax.set_ylabel(self.labels['speedup'], fontweight='bold')
        ax.set_title('GPU 수에 따른 속도 향상 / Speedup vs GPU Count',
                    fontweight='bold', pad=20)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, max_gpus + 0.5)

        self.save_figure(fig, '2_speedup_vs_gpus')

    def figure3_strong_scaling(self, df: pd.DataFrame):
        """
        그림 3: 강한 확장성 곡선
        Figure 3: Strong scaling curve
        """
        print("📊 그림 3: 강한 확장성 곡선 생성 중...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 왼쪽: 실행 시간
        for dataset in df['데이터셋 (Dataset)'].unique():
            data = df[df['데이터셋 (Dataset)'] == dataset].sort_values('GPU 수 (# GPUs)')
            if len(data) > 1:
                ax1.plot(data['GPU 수 (# GPUs)'], data['실행 시간 (ms)'],
                        marker='o', linewidth=2, markersize=8,
                        label=dataset.replace('.txt', ''))
                ax1.plot(data['GPU 수 (# GPUs)'], data['이상적 시간 (ms)'],
                        linestyle='--', alpha=0.5)

        ax1.set_xlabel(self.labels['gpu_count'], fontweight='bold')
        ax1.set_ylabel(self.labels['time'], fontweight='bold')
        ax1.set_title('실행 시간 / Execution Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 오른쪽: 효율
        for dataset in df['데이터셋 (Dataset)'].unique():
            data = df[df['데이터셋 (Dataset)'] == dataset].sort_values('GPU 수 (# GPUs)')
            if len(data) > 1:
                ax2.plot(data['GPU 수 (# GPUs)'], data['효율 (Efficiency, %)'],
                        marker='s', linewidth=2, markersize=8,
                        label=dataset.replace('.txt', ''))

        ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5,
                   label='100% 효율 / 100% Efficiency', alpha=0.5)
        ax2.set_xlabel(self.labels['gpu_count'], fontweight='bold')
        ax2.set_ylabel(self.labels['efficiency'], fontweight='bold')
        ax2.set_title('병렬 효율 / Parallel Efficiency', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 110)

        plt.tight_layout()
        self.save_figure(fig, '3_strong_scaling')

    def figure4_edge_cut_comparison(self, df: pd.DataFrame):
        """
        그림 4: 간선 절단 비교
        Figure 4: Edge-cut comparison
        """
        print("📊 그림 4: 간선 절단 비교 생성 중...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 알고리즘별 간선 절단
        algorithms = df['알고리즘 (Algorithm)'].unique()
        datasets = df['데이터셋 (Dataset)'].unique()[:5]

        x = np.arange(len(datasets))
        width = 0.35

        for i, algo in enumerate(algorithms):
            data = df[df['알고리즘 (Algorithm)'] == algo]
            edge_cuts = [data[data['데이터셋 (Dataset)'] == d]['간선 절단 (Edge-Cut)'].values[0]
                        if len(data[data['데이터셋 (Dataset)'] == d]) > 0 else 0
                        for d in datasets]

            offset = width * i
            color = COLORS.get(algo, f'C{i}')
            ax.bar(x + offset, edge_cuts, width, label=algo, color=color, alpha=0.8)

        ax.set_xlabel(self.labels['dataset'], fontweight='bold')
        ax.set_ylabel(self.labels['edge_cut'], fontweight='bold')
        ax.set_title('간선 절단 비교 / Edge-Cut Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([d.replace('.txt', '')[:15] for d in datasets],
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        self.save_figure(fig, '4_edge_cut_comparison')

    def figure5_communication_volume(self, df: pd.DataFrame):
        """
        그림 5: 통신량 분석
        Figure 5: Communication volume analysis
        """
        print("📊 그림 5: 통신량 분석 생성 중...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 왼쪽: 통신량
        algorithms = df['알고리즘 (Algorithm)'].unique()
        datasets = df['데이터셋 (Dataset)'].unique()[:5]

        x = np.arange(len(datasets))
        width = 0.35

        for i, algo in enumerate(algorithms):
            data = df[df['알고리즘 (Algorithm)'] == algo]
            volumes = [data[data['데이터셋 (Dataset)'] == d]['통신량 (MB)'].values[0]
                      if len(data[data['데이터셋 (Dataset)'] == d]) > 0 else 0
                      for d in datasets]

            offset = width * i
            ax1.bar(x + offset, volumes, width, label=algo, alpha=0.8)

        ax1.set_xlabel(self.labels['dataset'], fontweight='bold')
        ax1.set_ylabel(self.labels['communication'], fontweight='bold')
        ax1.set_title('통신량 / Communication Volume', fontweight='bold')
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels([d.replace('.txt', '')[:12] for d in datasets],
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 오른쪽: 대역폭 활용
        for algo in algorithms:
            data = df[df['알고리즘 (Algorithm)'] == algo]
            if len(data) > 0 and '대역폭 (GB/s)' in data.columns:
                bandwidths = data.groupby('GPU 수 (# GPUs)')['대역폭 (GB/s)'].mean()
                ax2.plot(bandwidths.index, bandwidths.values,
                        marker='o', linewidth=2, markersize=8, label=algo)

        ax2.set_xlabel(self.labels['gpu_count'], fontweight='bold')
        ax2.set_ylabel(self.labels['bandwidth'], fontweight='bold')
        ax2.set_title('대역폭 활용 / Bandwidth Utilization', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_figure(fig, '5_communication_volume')

    def figure6_memory_usage(self, df: pd.DataFrame):
        """
        그림 6: 메모리 사용량
        Figure 6: Memory usage
        """
        print("📊 그림 6: 메모리 사용량 생성 중...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 알고리즘별 메모리 사용량
        for algo in df['알고리즘 (Algorithm)'].unique():
            data = df[df['알고리즘 (Algorithm)'] == algo].sort_values('간선 수 (Edges)')
            if len(data) > 2:
                ax.plot(data['간선 수 (Edges)'] / 1e6, data['총 메모리 (MB)'],
                       marker='o', linewidth=2, markersize=8, label=algo)

        ax.set_xlabel('간선 수 (백만) / Edges (millions)', fontweight='bold')
        ax.set_ylabel(self.labels['memory'], fontweight='bold')
        ax.set_title('메모리 사용량 / Memory Usage', fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

        self.save_figure(fig, '6_memory_usage')

    def figure7_throughput_comparison(self, df: pd.DataFrame):
        """
        그림 7: 처리량 (MTEPS) 비교
        Figure 7: Throughput (MTEPS) comparison
        """
        print("📊 그림 7: 처리량 비교 생성 중...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 알고리즘별 처리량
        algorithms = df['알고리즘 (Algorithm)'].unique()
        x = np.arange(len(algorithms))

        throughputs = [df[df['알고리즘 (Algorithm)'] == algo]['처리량 (MTEPS)'].mean()
                      for algo in algorithms]
        colors = [COLORS.get(algo, f'C{i}') for i, algo in enumerate(algorithms)]

        bars = ax.bar(x, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # 막대 위에 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(self.labels['throughput'], fontweight='bold')
        ax.set_title('처리량 비교 (평균) / Average Throughput Comparison',
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        self.save_figure(fig, '7_throughput_comparison')

    def figure8_scalability_efficiency(self, df: pd.DataFrame):
        """
        그림 8: 확장성 효율
        Figure 8: Scalability efficiency
        """
        print("📊 그림 8: 확장성 효율 생성 중...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # GPU 수별 효율
        for algo in df['알고리즘 (Algorithm)'].unique():
            data = df[df['알고리즘 (Algorithm)'] == algo].groupby('GPU 수 (# GPUs)')['효율 (Efficiency, %)'].mean()
            if len(data) > 1:
                ax.plot(data.index, data.values,
                       marker='o', linewidth=2.5, markersize=10, label=algo)

        # 참조선들
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5,
                  label='100% (이상적)', alpha=0.6)
        ax.axhline(y=80, color='orange', linestyle=':', linewidth=1.5,
                  label='80% (우수)', alpha=0.6)
        ax.axhline(y=60, color='red', linestyle=':', linewidth=1.5,
                  label='60% (양호)', alpha=0.6)

        ax.set_xlabel(self.labels['gpu_count'], fontweight='bold')
        ax.set_ylabel(self.labels['efficiency'], fontweight='bold')
        ax.set_title('병렬 확장성 효율 / Parallel Scalability Efficiency',
                    fontweight='bold', pad=20)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)

        self.save_figure(fig, '8_scalability_efficiency')

    def generate_all_figures(self, figures: List[str] = None):
        """
        모든 그래프 생성 / Generate all figures

        Args:
            figures: 생성할 그래프 이름 리스트 (None이면 전체)
        """
        print("\n" + "=" * 80)
        print("📊 논문 그래프 생성 시작 / Starting Paper Figure Generation")
        print("=" * 80 + "\n")

        # CSV 파일 로드
        perf_df = None
        speedup_df = None
        comm_df = None
        scale_df = None
        mem_df = None

        try:
            perf_path = self.data_dir / 'performance_summary.csv'
            if perf_path.exists():
                perf_df = pd.read_csv(perf_path)
                print(f"✅ 성능 데이터 로드: {len(perf_df)} 행")

            speedup_path = self.data_dir / 'speedup_table.csv'
            if speedup_path.exists():
                speedup_df = pd.read_csv(speedup_path)
                print(f"✅ 속도 향상 데이터 로드: {len(speedup_df)} 행")

            comm_path = self.data_dir / 'communication_analysis.csv'
            if comm_path.exists():
                comm_df = pd.read_csv(comm_path)
                print(f"✅ 통신 데이터 로드: {len(comm_df)} 행")

            scale_path = self.data_dir / 'scalability_data.csv'
            if scale_path.exists():
                scale_df = pd.read_csv(scale_path)
                print(f"✅ 확장성 데이터 로드: {len(scale_df)} 행")

            mem_path = self.data_dir / 'memory_usage.csv'
            if mem_path.exists():
                mem_df = pd.read_csv(mem_path)
                print(f"✅ 메모리 데이터 로드: {len(mem_df)} 행")

        except Exception as e:
            print(f"⚠️ 데이터 로드 오류: {e}")
            return

        print()

        # 그래프 생성 매핑
        figure_functions = {
            'execution_time': (self.figure1_execution_time_comparison, perf_df),
            'speedup': (self.figure2_speedup_vs_gpus, scale_df),
            'scaling': (self.figure3_strong_scaling, scale_df),
            'edge_cut': (self.figure4_edge_cut_comparison, comm_df),
            'communication': (self.figure5_communication_volume, comm_df),
            'memory': (self.figure6_memory_usage, mem_df),
            'throughput': (self.figure7_throughput_comparison, perf_df),
            'efficiency': (self.figure8_scalability_efficiency, scale_df)
        }

        # 생성할 그래프 결정
        if figures is None:
            figures = list(figure_functions.keys())

        # 각 그래프 생성
        for fig_name in figures:
            if fig_name in figure_functions:
                func, df = figure_functions[fig_name]
                if df is not None and len(df) > 0:
                    try:
                        func(df)
                    except Exception as e:
                        print(f"  ⚠️ 오류 발생: {e}")
                else:
                    print(f"  ⚠️ 데이터 없음: {fig_name}")
            else:
                print(f"  ⚠️ 알 수 없는 그래프: {fig_name}")

        print("\n" + "=" * 80)
        print("🎉 모든 그래프 생성 완료! / All figures generated successfully!")
        print("=" * 80)

def main():
    """메인 함수 / Main function"""
    parser = argparse.ArgumentParser(
        description='논문용 그래프 생성 / Generate figures for paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제 / Examples:
  python generate_paper_figures.py --data paper_results/ --output figures/
  python generate_paper_figures.py -d results/processed -o figures/ --language english
  python generate_paper_figures.py --figures speedup,scaling --dpi 600
        """
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        default='results/processed',
        help='CSV 데이터 디렉토리 / CSV data directory'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='figures',
        help='그래프 출력 디렉토리 / Figure output directory'
    )

    parser.add_argument(
        '--language', '-l',
        type=str,
        choices=['korean', 'english'],
        default='korean',
        help='그래프 언어 / Figure language'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='해상도 (DPI) / Resolution (DPI)'
    )

    parser.add_argument(
        '--figures', '-f',
        type=str,
        default=None,
        help='생성할 그래프 (쉼표 구분) / Figures to generate (comma-separated)'
    )

    args = parser.parse_args()

    # 생성할 그래프 파싱
    figures_to_generate = None
    if args.figures:
        figures_to_generate = [f.strip() for f in args.figures.split(',')]

    # 그래프 생성기 생성
    generator = PaperFigureGenerator(
        args.data,
        args.output,
        language=args.language,
        dpi=args.dpi
    )

    # 모든 그래프 생성
    generator.generate_all_figures(figures=figures_to_generate)

if __name__ == '__main__':
    main()
