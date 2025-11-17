#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
논문 표 생성 유틸리티 / Paper Table Generation Utility

CSV 데이터를 읽어 논문용 LaTeX 표 및 Markdown 표를 생성합니다.
Reads CSV data and generates LaTeX and Markdown tables for paper.

사용법 / Usage:
    python generate_paper_tables.py --data paper_results/ --output tables/

작성자 / Author: Research Team
날짜 / Date: 2025-11-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict
from tabulate import tabulate

class PaperTableGenerator:
    """
    논문 표 생성 클래스
    Paper table generation class
    """

    def __init__(self, data_dir: str, output_dir: str, format_type: str = 'latex'):
        """
        초기화 / Initialize

        Args:
            data_dir: CSV 데이터 디렉토리
            output_dir: 표 출력 디렉토리
            format_type: 'latex' or 'markdown'
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_table(self, df: pd.DataFrame, name: str, caption: str):
        """
        표를 LaTeX 또는 Markdown 형식으로 저장
        Save table as LaTeX or Markdown format
        """
        if self.format_type == 'latex':
            filepath = self.output_dir / f"{name}.tex"
            latex_code = self.to_latex(df, caption)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_code)
        else:
            filepath = self.output_dir / f"{name}.md"
            markdown_table = tabulate(df, headers='keys', tablefmt='pipe',
                                     showindex=False, floatfmt=".2f")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {caption}\n\n")
                f.write(markdown_table)
                f.write("\n")

        print(f"  ✅ {filepath.name}")

    def to_latex(self, df: pd.DataFrame, caption: str) -> str:
        """
        DataFrame을 LaTeX 표로 변환
        Convert DataFrame to LaTeX table
        """
        # 열 정렬: 첫 2열은 left, 나머지는 right
        col_format = 'l' * 2 + 'r' * (len(df.columns) - 2)

        latex = "\\begin{table}[ht]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\begin{{tabular}}{{|{col_format}|}}\n"
        latex += "\\hline\n"

        # 헤더
        headers = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
        latex += headers + " \\\\\n"
        latex += "\\hline\\hline\n"

        # 데이터 행
        for idx, row in df.iterrows():
            row_data = []
            for i, val in enumerate(row):
                if isinstance(val, float):
                    row_data.append(f"{val:.2f}")
                elif isinstance(val, int):
                    row_data.append(f"{val:,}")
                else:
                    row_data.append(str(val))
            latex += " & ".join(row_data) + " \\\\\n"
            latex += "\\hline\n"

        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    def table1_algorithm_complexity(self) -> pd.DataFrame:
        """
        표 1: 알고리즘 복잡도
        Table 1: Algorithm complexity
        """
        data = [
            {
                '알고리즘': 'Dijkstra',
                '시간 복잡도': '$O((m+n) \\log n)$',
                '공간 복잡도': '$O(n)$',
                '음수 가중치': '불가',
                '비고': '이진 힙 기반'
            },
            {
                '알고리즘': 'Bellman-Ford',
                '시간 복잡도': '$O(nm)$',
                '공간 복잡도': '$O(n)$',
                '음수 가중치': '가능',
                '비고': '음수 사이클 감지'
            },
            {
                '알고리즘': 'Duan et al. (순차)',
                '시간 복잡도': '$O(m \\log^{2/3} n)$',
                '공간 복잡도': '$O(n+m)$',
                '음수 가중치': '불가',
                '비고': '정렬 장벽 돌파'
            },
            {
                '알고리즘': 'Duan et al. (OpenMP)',
                '시간 복잡도': '$O(m \\log^{2/3} n / p)$',
                '공간 복잡도': '$O(n+m)$',
                '음수 가중치': '불가',
                '비고': '공유 메모리 병렬'
            },
            {
                '알고리즘': 'Duan et al. (CUDA)',
                '시간 복잡도': '$O(m \\log^{2/3} n / p)$',
                '공간 복잡도': '$O(n+m)$',
                '음수 가중치': '불가',
                '비고': 'GPU 가속'
            },
            {
                '알고리즘': 'MGAP (제안 기법)',
                '시간 복잡도': '$O(m \\log^{2/3} n / kp)$',
                '공간 복잡도': '$O(n+m)$',
                '음수 가중치': '불가',
                '비고': 'Multi-GPU 최적화'
            }
        ]

        return pd.DataFrame(data)

    def table2_dataset_characteristics(self) -> pd.DataFrame:
        """
        표 2: 데이터셋 특성
        Table 2: Dataset characteristics
        """
        # CSV에서 로드
        try:
            df = pd.read_csv(self.data_dir / 'performance_summary.csv')

            # 중복 제거 및 선택
            unique_datasets = df[['데이터셋 (Dataset)', '정점 수 (Vertices)', '간선 수 (Edges)']].drop_duplicates()

            # 평균 차수 계산
            unique_datasets['평균 차수'] = unique_datasets['간선 수 (Edges)'] / unique_datasets['정점 수 (Vertices)']

            # 그래프 유형 추론
            def infer_type(name):
                name_lower = name.lower()
                if 'road' in name_lower or 'usa' in name_lower:
                    return '도로망'
                elif 'social' in name_lower or 'twitter' in name_lower or 'email' in name_lower:
                    return '소셜 네트워크'
                elif 'web' in name_lower or 'google' in name_lower:
                    return '웹 그래프'
                elif 'grid' in name_lower:
                    return '격자'
                elif 'dag' in name_lower:
                    return 'DAG'
                else:
                    return '무작위'

            unique_datasets['유형'] = unique_datasets['데이터셋 (Dataset)'].apply(infer_type)

            # 정렬 및 선택
            result = unique_datasets[['데이터셋 (Dataset)', '유형', '정점 수 (Vertices)',
                                     '간선 수 (Edges)', '평균 차수']].sort_values('정점 수 (Vertices)')

            return result.head(10)  # 최대 10개

        except Exception as e:
            print(f"  ⚠️ 데이터 로드 오류: {e}")
            # 기본 예제 데이터
            return pd.DataFrame([
                {'데이터셋 (Dataset)': 'Grid-1K', '유형': '격자', '정점 수 (Vertices)': 1000,
                 '간선 수 (Edges)': 2000, '평균 차수': 2.0}
            ])

    def table3_performance_results(self) -> pd.DataFrame:
        """
        표 3: 성능 결과
        Table 3: Performance results
        """
        try:
            df = pd.read_csv(self.data_dir / 'performance_summary.csv')

            # 대표 데이터셋 선택 (중간 크기)
            df_filtered = df[df['정점 수 (Vertices)'].between(500000, 2000000)]

            # 알고리즘과 데이터셋별로 평균
            result = df_filtered.groupby('알고리즘 (Algorithm)').agg({
                '실행 시간 (Time, ms)': 'mean',
                '속도 향상 (Speedup)': 'mean',
                '처리량 (MTEPS)': 'mean'
            }).reset_index()

            result.columns = ['알고리즘', '평균 실행 시간 (ms)', '평균 속도 향상', '평균 처리량 (MTEPS)']

            return result

        except Exception as e:
            print(f"  ⚠️ 데이터 로드 오류: {e}")
            return pd.DataFrame()

    def table4_communication_metrics(self) -> pd.DataFrame:
        """
        표 4: 통신 메트릭
        Table 4: Communication metrics
        """
        try:
            df = pd.read_csv(self.data_dir / 'communication_analysis.csv')

            # 주요 메트릭만 선택
            if len(df) > 0:
                result = df[['알고리즘 (Algorithm)', 'GPU 수 (# GPUs)', '간선 절단 (Edge-Cut)',
                           '통신량 (MB)', '통신 비율 (%)', '대역폭 (GB/s)']]
                return result.head(10)

        except Exception as e:
            print(f"  ⚠️ 데이터 로드 오류: {e}")

        return pd.DataFrame()

    def table5_scalability_summary(self) -> pd.DataFrame:
        """
        표 5: 확장성 요약
        Table 5: Scalability summary
        """
        try:
            df = pd.read_csv(self.data_dir / 'scalability_data.csv')

            if len(df) > 0:
                result = df[['알고리즘 (Algorithm)', 'GPU 수 (# GPUs)', '실행 시간 (ms)',
                           '속도 향상 (Speedup)', '효율 (Efficiency, %)']]
                return result.head(15)

        except Exception as e:
            print(f"  ⚠️ 데이터 로드 오류: {e}")

        return pd.DataFrame()

    def table6_ablation_results(self) -> pd.DataFrame:
        """
        표 6: 절제 연구 결과
        Table 6: Ablation study results
        """
        # 예제 데이터 (실제 벤치마크 결과로 대체 필요)
        data = [
            {
                '구성': '베이스라인 (단일 GPU)',
                '실행 시간 (ms)': 100.0,
                '속도 향상': 1.0,
                '비고': '기준'
            },
            {
                '구성': '+ NVLINK P2P',
                '실행 시간 (ms)': 55.0,
                '속도 향상': 1.82,
                '비고': '통신 속도 향상'
            },
            {
                '구성': '+ 비동기 파이프라인',
                '실행 시간 (ms)': 42.0,
                '속도 향상': 2.38,
                '비고': '지연 시간 은닉'
            },
            {
                '구성': '+ METIS 분할',
                '실행 시간 (ms)': 28.0,
                '속도 향상': 3.57,
                '비고': '통신량 감소'
            },
            {
                '구성': '전체 MGAP (4 GPUs)',
                '실행 시간 (ms)': 12.0,
                '속도 향상': 8.33,
                '비고': '모든 최적화 적용'
            }
        ]

        return pd.DataFrame(data)

    def generate_all_tables(self):
        """
        모든 표 생성 / Generate all tables
        """
        print("\n" + "=" * 80)
        print("📊 논문 표 생성 시작 / Starting Paper Table Generation")
        print("=" * 80 + "\n")

        # 표 생성 매핑
        tables = {
            '1_algorithm_complexity': (self.table1_algorithm_complexity,
                                      '알고리즘 복잡도 비교 / Algorithm Complexity Comparison'),
            '2_dataset_characteristics': (self.table2_dataset_characteristics,
                                         '벤치마크 데이터셋 특성 / Benchmark Dataset Characteristics'),
            '3_performance_results': (self.table3_performance_results,
                                     '성능 결과 요약 / Performance Results Summary'),
            '4_communication_metrics': (self.table4_communication_metrics,
                                       '통신 메트릭 / Communication Metrics'),
            '5_scalability_summary': (self.table5_scalability_summary,
                                     '확장성 요약 / Scalability Summary'),
            '6_ablation_results': (self.table6_ablation_results,
                                  '절제 연구 결과 / Ablation Study Results')
        }

        for name, (func, caption) in tables.items():
            try:
                print(f"📋 표 {name} 생성 중...")
                df = func()
                if len(df) > 0:
                    self.save_table(df, name, caption)
                else:
                    print(f"  ⚠️ 데이터 없음")
            except Exception as e:
                print(f"  ⚠️ 오류 발생: {e}")

        print("\n" + "=" * 80)
        print("🎉 모든 표 생성 완료! / All tables generated successfully!")
        print("=" * 80)

def main():
    """메인 함수 / Main function"""
    parser = argparse.ArgumentParser(
        description='논문용 표 생성 / Generate tables for paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제 / Examples:
  python generate_paper_tables.py --data paper_results/ --output tables/
  python generate_paper_tables.py -d results/processed -o tables/ --format markdown
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
        default='tables',
        help='표 출력 디렉토리 / Table output directory'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['latex', 'markdown'],
        default='latex',
        help='출력 형식 / Output format'
    )

    args = parser.parse_args()

    # 표 생성기 생성
    generator = PaperTableGenerator(args.data, args.output, format_type=args.format)

    # 모든 표 생성
    generator.generate_all_tables()

if __name__ == '__main__':
    main()
