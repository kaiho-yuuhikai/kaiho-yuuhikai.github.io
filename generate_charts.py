#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3回 開邦高校大同窓会 データ分析レポート用グラフ生成
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 日本語フォント設定
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# カラーパレット
COLORS = {
    'primary': '#1A365D',      # ダークブルー
    'secondary': '#3182CE',    # ライトブルー
    'accent': '#E53E3E',       # レッド
    'success': '#38A169',      # グリーン
    'warning': '#D69E2E',      # イエロー
    'gray': '#718096',         # グレー
    'light_gray': '#E2E8F0',   # ライトグレー
}

# グラデーションカラー
GRADIENT_BLUES = ['#1A365D', '#2C5282', '#2B6CB0', '#3182CE', '#4299E1', '#63B3ED', '#90CDF4']

def setup_style():
    """グラフスタイルの設定"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = COLORS['gray']
    plt.rcParams['grid.color'] = COLORS['light_gray']
    plt.rcParams['axes.labelcolor'] = COLORS['primary']
    plt.rcParams['xtick.color'] = COLORS['primary']
    plt.rcParams['ytick.color'] = COLORS['primary']

def save_figure(fig, filename):
    """図を保存"""
    filepath = f'images/analysis/{filename}'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {filepath}')

# ===========================================
# 1. 期別参加者数の棒グラフ
# ===========================================
def chart_participants_by_class():
    """期別参加者数"""
    classes = list(range(1, 37))
    participants = [30, 32, 45, 26, 6, 7, 19, 7, 16, 21,
                    23, 28, 27, 10, 25, 13, 5, 19, 9, 28,
                    16, 10, 11, 11, 4, 20, 16, 5, 4, 7,
                    6, 2, 10, 5, 3, 7]

    fig, ax = plt.subplots(figsize=(14, 6))

    # 世代別に色分け
    colors = []
    for i, c in enumerate(classes):
        if c <= 10:
            colors.append(COLORS['primary'])
        elif c <= 20:
            colors.append(COLORS['secondary'])
        elif c <= 30:
            colors.append('#63B3ED')
        else:
            colors.append('#90CDF4')

    bars = ax.bar(classes, participants, color=colors, edgecolor='white', linewidth=0.5)

    # 最大値をハイライト
    max_idx = participants.index(max(participants))
    bars[max_idx].set_color(COLORS['accent'])

    ax.set_xlabel('期', fontsize=12, fontweight='bold')
    ax.set_ylabel('参加者数（名）', fontsize=12, fontweight='bold')
    ax.set_title('期別参加者数', fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(range(1, 37, 2))
    ax.set_xlim(0, 37)
    ax.set_ylim(0, 50)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='1〜10期（ベテラン）'),
        Patch(facecolor=COLORS['secondary'], label='11〜20期（中堅）'),
        Patch(facecolor='#63B3ED', label='21〜30期'),
        Patch(facecolor='#90CDF4', label='31〜37期（若手）'),
        Patch(facecolor=COLORS['accent'], label='最多（3期: 45名）'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 平均線
    avg = np.mean(participants)
    ax.axhline(y=avg, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.7)
    ax.text(35.5, avg + 1, f'平均: {avg:.1f}名', fontsize=10, color=COLORS['warning'], ha='right')

    save_figure(fig, 'participants_by_class.png')

# ===========================================
# 2. 世代グループ別参加状況（円グラフ）
# ===========================================
def chart_generation_pie():
    """世代グループ別参加状況"""
    labels = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']
    sizes = [209, 187, 104, 33]
    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4']
    explode = (0.02, 0.02, 0.02, 0.05)

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))}名)',
        startangle=90,
        pctdistance=0.6,
        labeldistance=1.15,
        textprops={'fontsize': 11}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax.set_title('世代グループ別参加状況', fontsize=16, fontweight='bold', pad=20)

    # 中央にテキスト
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0.05, '総参加者', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['gray'])
    ax.text(0, -0.12, '533名', ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['primary'])

    save_figure(fig, 'generation_distribution.png')

# ===========================================
# 3. 満足度評価（横棒グラフ）
# ===========================================
def chart_satisfaction():
    """満足度評価"""
    categories = ['時間帯\n(15:00-17:30)', '日程\n(12/28)', '会費妥当性', '料理・ドリンク']
    scores = [4.29, 4.05, 3.90, 3.34]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 色を満足度に応じて変更
    colors = []
    for s in scores:
        if s >= 4.0:
            colors.append(COLORS['success'])
        elif s >= 3.5:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['accent'])

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor='white')

    # スコアをバーの右に表示
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=12, fontweight='bold',
                color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel('満足度スコア（5段階評価）', fontsize=12, fontweight='bold')
    ax.set_title('満足度評価（項目別）', fontsize=16, fontweight='bold', pad=20)

    # 目標線
    ax.axvline(x=4.0, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(4.05, 3.7, '目標: 4.0', fontsize=9, color=COLORS['gray'])

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='良好（4.0以上）'),
        Patch(facecolor=COLORS['warning'], label='普通（3.5〜4.0）'),
        Patch(facecolor=COLORS['accent'], label='要改善（3.5未満）'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.invert_yaxis()

    save_figure(fig, 'satisfaction_scores.png')

# ===========================================
# 4. 情報入手経路（円グラフ）
# ===========================================
def chart_information_source():
    """情報入手経路"""
    labels = ['同窓生からの\n口コミ', '雄飛会\nFacebook', '大同窓会\nInstagram', 'ポスター', 'その他']
    sizes = [83.1, 23.7, 13.6, 5.1, 5.0]
    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4', COLORS['light_gray']]

    fig, ax = plt.subplots(figsize=(9, 7))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.15,
        textprops={'fontsize': 10}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('情報入手経路（複数回答）', fontsize=16, fontweight='bold', pad=20)

    save_figure(fig, 'information_source.png')

# ===========================================
# 5. 不参加理由（横棒グラフ）
# ===========================================
def chart_non_participation_reasons():
    """不参加理由"""
    reasons = ['仕事の都合', '県外・海外在住', '広報不足', '土曜日希望', '家庭の事情', '会費が高い']
    percentages = [39.0, 35.6, 20.3, 8.5, 15.0, 6.8]

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(reasons))
    bars = ax.barh(y_pos, percentages, color=COLORS['secondary'], height=0.6, edgecolor='white')

    # 最も多い理由をハイライト
    bars[0].set_color(COLORS['primary'])
    bars[1].set_color(COLORS['primary'])

    for bar, pct in zip(bars, percentages):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(reasons, fontsize=11)
    ax.set_xlim(0, 50)
    ax.set_xlabel('割合（%）', fontsize=12, fontweight='bold')
    ax.set_title('周囲の不参加理由（複数回答）', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'non_participation_reasons.png')

# ===========================================
# 6. 希望プログラム（横棒グラフ）
# ===========================================
def chart_desired_programs():
    """希望プログラム"""
    programs = [
        '校歌斉唱\n（芸術科合唱つき）',
        '思い出ビデオ\n・スライドショー',
        '学科・専門分野別\n交流コーナー',
        '在校生の\n活動紹介',
        '卒業生有志の\n音楽・パフォーマンス',
        'スマホ参加型\n企画（クイズ等）'
    ]
    percentages = [49.2, 47.5, 42.4, 40.7, 35.6, 32.2]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(programs))
    colors = [COLORS['primary'] if p >= 45 else COLORS['secondary'] for p in percentages]
    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor='white')

    for bar, pct in zip(bars, percentages):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(programs, fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_xlabel('希望率（%）', fontsize=12, fontweight='bold')
    ax.set_title('次回取り入れてほしいプログラム TOP6', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'desired_programs.png')

# ===========================================
# 7. 会費許容額の分布
# ===========================================
def chart_fee_tolerance():
    """会費許容額の世代別分布"""
    generations = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']
    current_fee = [5500, 5500, 5500, 3000]
    tolerance = [6176, 6167, 5800, 5000]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(generations))
    width = 0.35

    bars1 = ax.bar(x - width/2, current_fee, width, label='現行会費', color=COLORS['gray'], edgecolor='white')
    bars2 = ax.bar(x + width/2, tolerance, width, label='許容額平均', color=COLORS['secondary'], edgecolor='white')

    # 差額を表示
    for i, (curr, tol) in enumerate(zip(current_fee, tolerance)):
        diff = tol - curr
        if diff > 0:
            ax.annotate(f'+{diff:,}円', xy=(i + width/2, tol + 100),
                       ha='center', fontsize=10, fontweight='bold', color=COLORS['success'])

    ax.set_ylabel('金額（円）', fontsize=12, fontweight='bold')
    ax.set_title('世代別 会費許容額 vs 現行会費', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(generations, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 7500)

    # 値を表示
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'¥{int(bar.get_height()):,}', ha='center', fontsize=9, color=COLORS['gray'])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'¥{int(bar.get_height()):,}', ha='center', fontsize=9, color=COLORS['secondary'])

    save_figure(fig, 'fee_tolerance.png')

# ===========================================
# 8. 協力意欲の高い層の分析
# ===========================================
def chart_cooperation_willingness():
    """協力内容の傾向"""
    items = [
        '寄付や協賛\nによる支援',
        '特別授業等の\n講師',
        '具体的な相談が\nあれば検討',
        '次回の\n実行委員',
        '広報協力\n（SNS等）',
        '雄飛会の\n役員活動'
    ]
    counts = [17, 14, 10, 6, 6, 3]
    percentages = [63.0, 51.9, 37.0, 22.2, 22.2, 11.1]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(items))
    colors = [COLORS['primary'] if p >= 50 else COLORS['secondary'] if p >= 30 else '#63B3ED' for p in percentages]
    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor='white')

    for bar, pct, cnt in zip(bars, percentages, counts):
        ax.text(pct + 2, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%（{cnt}名）', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(items, fontsize=10)
    ax.set_xlim(0, 80)
    ax.set_xlabel('割合（%）/ 意欲の高い27名中', fontsize=12, fontweight='bold')
    ax.set_title('協力内容の傾向（取組意欲の高い層）', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'cooperation_willingness.png')

# ===========================================
# 9. 運営関与と参加者数の関係
# ===========================================
def chart_involvement_effect():
    """運営関与タイプと参加者数"""
    types = ['メンター＋運営\n両方', '本編登壇者\nあり', 'メンター\nのみ', '運営のみ', '関与なし\n（1-30期）', '関与なし\n（31-36期）']
    avg_participants = [23.8, 16.2, 16.0, 11.4, 9.0, 3.7]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(types))
    colors = [COLORS['success'] if p >= 20 else COLORS['secondary'] if p >= 15 else COLORS['warning'] if p >= 10 else COLORS['accent'] for p in avg_participants]
    bars = ax.bar(x, avg_participants, color=colors, edgecolor='white', width=0.6)

    for bar, val in zip(bars, avg_participants):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}名', ha='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.set_ylabel('平均参加者数（名/期）', fontsize=12, fontweight='bold')
    ax.set_title('運営関与タイプ別 平均参加者数', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 30)

    # 差を示す矢印
    ax.annotate('', xy=(0, 23.8), xytext=(4, 9.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.text(2.5, 18, '約2.6倍の差', fontsize=11, fontweight='bold', color=COLORS['accent'], ha='center')

    save_figure(fig, 'involvement_effect.png')

# ===========================================
# メイン処理
# ===========================================
if __name__ == '__main__':
    print('Generating charts...')
    setup_style()

    chart_participants_by_class()
    chart_generation_pie()
    chart_satisfaction()
    chart_information_source()
    chart_non_participation_reasons()
    chart_desired_programs()
    chart_fee_tolerance()
    chart_cooperation_willingness()
    chart_involvement_effect()

    print('All charts generated successfully!')
