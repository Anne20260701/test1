import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd

import matplotlib.font_manager as fm
import os
font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")


st.set_page_config(
    page_title="神积脑盾 - 实时分析引擎",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 神积脑盾 - 实时EEG/ERP分析引擎")
st.caption("基于听觉Oddball范式的P300检测与保险反欺诈评分")

# ============================================
# 核心信号处理函数（真正在干活的部分）
# ============================================

def generate_eeg_data(duration=10, fs=256, amplitude=20, latency=300, noise_level=0.5, is_abnormal=False):
    """
    生成模拟EEG数据（含ERP成分）
    这是实际数据分析流水线的第一步：模拟数据输入
    """
    t = np.linspace(0, duration, int(duration * fs))
    
    # 背景EEG（Alpha节律 + 随机噪声）
    alpha = 5 * np.sin(2 * np.pi * 10 * t)
    theta = 2 * np.sin(2 * np.pi * 6 * t)
    noise = np.random.normal(0, noise_level, len(t))
    background = alpha + theta + noise
    
    # ERP事件：在特定时间点呈现刺激
    # 模拟Oddball范式：每隔1.5秒出现一个目标刺激
    erp_signal = np.zeros(len(t))
    stimulus_times = np.arange(1.5, duration, 1.5)
    
    for stim_time in stimulus_times:
        idx = int(stim_time * fs)
        if idx < len(t) - 200:
            # P300波形（使用更真实的ERP形状）
            erp_window = np.zeros(500)
            # P300：用更真实的波形
            p300 = amplitude * np.exp(-((np.arange(500) - 200) ** 2) / (2 * 50 ** 2))
            # N100
            n100 = -amplitude * 0.5 * np.exp(-((np.arange(500) - 100) ** 2) / (2 * 30 ** 2))
            # P200
            p200 = amplitude * 0.3 * np.exp(-((np.arange(500) - 180) ** 2) / (2 * 35 ** 2))
            erp_window = p300 + n100 + p200
            
            # 如果是异常情况（脑震荡），振幅降低、潜伏期延长
            if is_abnormal:
                erp_window = erp_window * 0.7  # 振幅降低30%
                # 潜伏期延长：整体右移
                shift = 30
                erp_window = np.roll(erp_window, shift)
                erp_window[:shift] = 0
            
            end_idx = min(idx + 500, len(t))
            erp_signal[idx:end_idx] += erp_window[:end_idx - idx]
    
    eeg_data = background + erp_signal * 0.3
    return t, eeg_data, stimulus_times, fs

def preprocess_eeg(eeg_data, fs):
    """
    预处理流水线：滤波、去趋势、归一化
    这是实际数据分析的第二步
    """
    # 带通滤波 0.5-30Hz（去除DC漂移和高频噪声）
    nyquist = fs / 2
    b, a = butter(4, [0.5 / nyquist, 30 / nyquist], btype='band')
    filtered = filtfilt(b, a, eeg_data)
    
    # 去趋势（去除缓慢漂移）
    filtered = signal.detrend(filtered)
    
    # 归一化
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    
    return normalized

def extract_erp(eeg_data, stimulus_times, fs, pre_stim=0.2, post_stim=0.8):
    """
    提取ERP：从连续EEG中切割刺激前后的片段并平均
    这是实际数据分析的第三步——提取ERP成分
    """
    pre_samples = int(pre_stim * fs)
    post_samples = int(post_stim * fs)
    epoch_length = pre_samples + post_samples
    
    epochs = []
    for stim_time in stimulus_times:
        idx = int(stim_time * fs)
        if idx > pre_samples and idx + post_samples < len(eeg_data):
            epoch = eeg_data[idx - pre_samples:idx + post_samples]
            epochs.append(epoch)
    
    if len(epochs) == 0:
        return None, None
    
    # 基线校正：减去刺激前平均值
    epochs = np.array(epochs)
    baseline = np.mean(epochs[:, :pre_samples], axis=1, keepdims=True)
    epochs = epochs - baseline
    
    # 平均ERP
    avg_erp = np.mean(epochs, axis=0)
    
    return avg_erp, epochs

def detect_p300(avg_erp, fs, pre_stim=0.2):
    """
    检测P300振幅和潜伏期
    这是实际数据分析的第四步——特征提取
    """
    if avg_erp is None:
        return None, None, None
    
    # 搜索窗口：250-500ms（P300典型范围）
    pre_samples = int(pre_stim * fs)
    start_idx = int(0.25 * fs) + pre_samples
    end_idx = int(0.5 * fs) + pre_samples
    
    if end_idx >= len(avg_erp):
        end_idx = len(avg_erp) - 1
    
    search_window = avg_erp[start_idx:end_idx]
    
    # P300是最大正峰值
    p300_amplitude = np.max(search_window)
    p300_idx = np.argmax(search_window) + start_idx
    p300_latency = (p300_idx - pre_samples) / fs * 1000  # 转换为毫秒
    
    return p300_amplitude, p300_latency, avg_erp

def calculate_risk_score(p300_amplitude, p300_latency, complaint_severity):
    """
    计算AI-Fraud Score（行为-神经不匹配指数）
    这是实际数据分析的第五步——输出风险评分
    """
    # 正常参考值：振幅~20μV，潜伏期~300ms
    normal_amplitude = 20
    normal_latency = 300
    
    # 计算神经异常程度
    amplitude_abnormality = max(0, (normal_amplitude - p300_amplitude) / normal_amplitude) * 50
    latency_abnormality = max(0, (p300_latency - normal_latency) / normal_latency) * 50
    neural_abnormality = min(100, amplitude_abnormality + latency_abnormality)
    
    # 行为-神经不匹配 = 主诉严重程度 - 神经异常程度
    mismatch = max(0, complaint_severity - neural_abnormality * 0.8)
    
    # AI-Fraud Score (0-1000)
    fraud_score = min(1000, mismatch * 8 + neural_abnormality * 2)
    
    return fraud_score, mismatch, neural_abnormality

# ============================================
# 主界面
# ============================================

# 侧边栏：参数控制
st.sidebar.header("⚙️ 分析参数")

# 场景选择
scenario = st.sidebar.selectbox(
    "选择分析场景",
    ["健康人（基线）", "疑似脑震荡（异常）", "疑似装病（行为-神经不匹配）"]
)

# 主诉严重程度
complaint = st.sidebar.slider(
    "主诉严重程度（0-100）",
    min_value=0, max_value=100, value=50,
    help="0=无症状，100=极度严重"
)

# 噪声水平（模拟不同信号质量）
noise_level = st.sidebar.slider(
    "信号噪声水平", min_value=0.1, max_value=2.0, value=0.5, step=0.1
)

# 运行按钮
run_analysis = st.sidebar.button("🚀 运行分析", type="primary", use_container_width=True)

# ============================================
# 主界面布局
# ============================================

if run_analysis:
    with st.spinner("正在分析EEG信号..."):
        # 步骤1：根据场景生成数据
        if scenario == "健康人（基线）":
            is_abnormal = False
            amplitude_factor = 1.0
            latency_shift = 0
            status_label = "正常"
        elif scenario == "疑似脑震荡（异常）":
            is_abnormal = True
            amplitude_factor = 0.7
            latency_shift = 30
            status_label = "异常"
        else:  # 装病
            # 装病者：行为上报告严重，但神经反应接近正常
            is_abnormal = False  # 神经反应接近正常
            amplitude_factor = 0.9
            latency_shift = 10
            status_label = "装病嫌疑"
        
        # 生成数据
        base_amplitude = 20 * amplitude_factor
        base_latency = 300 + latency_shift
        
        # 加入随机变异
        actual_amplitude = base_amplitude + np.random.normal(0, 1.5)
        actual_amplitude = max(8, min(22, actual_amplitude))
        
        actual_latency = base_latency + np.random.normal(0, 10)
        actual_latency = max(260, min(380, actual_latency))
        
        t, raw_eeg, stim_times, fs = generate_eeg_data(
            duration=10, fs=256, 
            amplitude=actual_amplitude, 
            latency=actual_latency,
            noise_level=noise_level,
            is_abnormal=(scenario == "疑似脑震荡（异常）")
        )
        
        # 步骤2：预处理
        processed_eeg = preprocess_eeg(raw_eeg, fs)
        
        # 步骤3：提取ERP
        avg_erp, epochs = extract_erp(processed_eeg, stim_times, fs)
        
        # 步骤4：检测P300
        p300_amp, p300_lat, erp_waveform = detect_p300(avg_erp, fs)
        
        # 步骤5：计算风险评分
        if p300_amp is not None:
            fraud_score, mismatch, neural_abnormality = calculate_risk_score(
                p300_amp, p300_lat, complaint
            )
        else:
            fraud_score, mismatch, neural_abnormality = 500, 50, 50

# ============================================
# 结果显示
# ============================================

if run_analysis:
    
    # 显示分析状态
    st.success("✅ 分析完成")
    
    # 第一行：四个核心指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("P300振幅", f"{p300_amp:.2f} μV", 
                  delta=f"{p300_amp - 20:.2f}" if p300_amp else None)
    with col2:
        st.metric("P300潜伏期", f"{p300_lat:.0f} ms",
                  delta=f"{p300_lat - 300:.0f}" if p300_lat else None)
    with col3:
        st.metric("神经异常指数", f"{neural_abnormality:.0f}/100")
    with col4:
        st.metric("行为-神经不匹配", f"{mismatch:.0f}/100")
    
    st.markdown("---")
    
    # 第二行：评分与决策
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("🎯 AI-Fraud Score")
        
        # 显示大号评分
        st.markdown(f"<h1 style='text-align: center; color: {'red' if fraud_score > 850 else 'orange' if fraud_score > 600 else 'green'};'>{fraud_score:.0f} / 1000</h1>", unsafe_allow_html=True)
        
        # 进度条
        st.progress(fraud_score / 1000)
        
        # 决策
        if fraud_score > 850:
            st.error("🚨 建议：自动拒赔，移交法务反欺诈部门")
        elif fraud_score > 600:
            st.warning("⚠️ 建议：分期给付，3个月后随访EEG")
        else:
            st.success("✅ 建议：极速闪赔")
    
    with col_b:
        st.subheader("📊 风险评估详情")
        
        risk_data = {
            "维度": ["主诉严重程度", "神经异常程度", "不匹配指数"],
            "数值": [f"{complaint}/100", f"{neural_abnormality:.0f}/100", f"{mismatch:.0f}/100"],
            "状态": [
                "🔴 高" if complaint > 70 else "🟡 中" if complaint > 40 else "🟢 低",
                "🔴 异常" if neural_abnormality > 60 else "🟡 轻度" if neural_abnormality > 30 else "🟢 正常",
                "🔴 不匹配" if mismatch > 60 else "🟡 轻度不匹配" if mismatch > 30 else "🟢 匹配"
            ]
        }
        st.table(pd.DataFrame(risk_data))
    
    st.markdown("---")
    
    # 第三行：波形可视化
    st.subheader("📈 信号分析可视化")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # 子图1：原始EEG + 预处理后
    time_axis = t[:len(raw_eeg)]
    axes[0].plot(time_axis, raw_eeg, 'b-', alpha=0.7, label='原始EEG', linewidth=0.5)
    axes[0].plot(time_axis[:len(processed_eeg)], processed_eeg, 'r-', alpha=0.7, label='预处理后', linewidth=0.5)
    
    # 标记刺激事件
    for stim_time in stim_times:
        if stim_time < time_axis[-1]:
            axes[0].axvline(x=stim_time, color='green', linestyle='--', alpha=0.3, linewidth=1)
    
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('振幅 (μV)')
    axes[0].set_title('EEG信号预处理')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：ERP波形（带P300标记）
    if erp_waveform is not None:
        erp_time = np.linspace(-0.2, 0.8, len(erp_waveform))
        axes[1].plot(erp_time, erp_waveform, 'b-', linewidth=2, label='ERP平均波形')
        
        # 标记P300峰值
        if p300_amp is not None:
            p300_time = (p300_lat / 1000) - 0.2
            axes[1].plot(p300_time, p300_amp, 'ro', markersize=10, label=f'P300: {p300_amp:.2f}μV @ {p300_lat:.0f}ms')
            axes[1].axvline(x=p300_time, color='red', linestyle='--', alpha=0.5)
        
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('时间 (秒)')
        axes[1].set_ylabel('振幅 (μV)')
        axes[1].set_title('ERP波形 (刺激后0-800ms)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # 子图3：所有epoch叠加
    if epochs is not None and len(epochs) > 0:
        epoch_time = np.linspace(-0.2, 0.8, epochs.shape[1])
        for i in range(min(20, epochs.shape[0])):
            axes[2].plot(epoch_time, epochs[i], 'gray', alpha=0.3, linewidth=0.5)
        
        if erp_waveform is not None:
            axes[2].plot(epoch_time, erp_waveform, 'r-', linewidth=2, label='平均ERP')
        
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_xlabel('时间 (秒)')
        axes[2].set_ylabel('振幅 (μV)')
        axes[2].set_title('单次试验叠加 (灰色) 与平均ERP (红色)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 解释
    st.markdown("---")
    st.caption("💡 **分析说明**：绿色虚线标记了每个听觉刺激的呈现时刻。P300是刺激后约300ms出现的正向波峰，其振幅和潜伏期是评估脑功能状态的核心指标。")
    
    # 场景特定解释
    st.info(f"""
    📋 **当前场景分析：{scenario}**
    
    - **P300振幅**：{p300_amp:.2f} μV（正常参考：18-22 μV）
    - **P300潜伏期**：{p300_lat:.0f} ms（正常参考：280-320 ms）
    - **AI-Fraud Score**：{fraud_score:.0f}/1000
    
    {"🔴 检测到显著的行为-神经不匹配，建议人工复核" if mismatch > 50 else "🟢 行为与神经指标基本一致"}
    """)

else:
    # 未运行时的初始状态
    st.info("👈 请先在左侧调整参数，然后点击 **'运行分析'** 按钮")
    
    # 显示示例图
    st.subheader("📈 分析流程预览")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    
    # 示例ERP
    t_ex = np.linspace(-0.2, 0.8, 500)
    p300_ex = 20 * np.exp(-((t_ex - 0.3) ** 2) / (2 * 0.05 ** 2))
    n100_ex = -10 * np.exp(-((t_ex - 0.1) ** 2) / (2 * 0.03 ** 2))
    p200_ex = 8 * np.exp(-((t_ex - 0.18) ** 2) / (2 * 0.035 ** 2))
    erp_ex = p300_ex + n100_ex + p200_ex
    
    axes[0].plot(t_ex, erp_ex, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='刺激呈现')
    axes[0].axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='P300 (~300ms)')
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('振幅 (μV)')
    axes[0].set_title('ERP波形示例（P300成分）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 决策流
    axes[1].axis('off')
    decision_text = """
    分析流程：
    原始EEG → 滤波去噪 → 切割epoch → 平均ERP → 检测P300 → 计算风险评分
    
    输出：AI-Fraud Score (0-1000)
    
    决策：
    0-600    → 极速闪赔
    600-850  → 分期给付 + 3个月随访
    850-1000 → 自动拒赔 + 移交法务
    """
    axes[1].text(0.1, 0.5, decision_text, fontsize=14, verticalalignment='center')
    
    plt.tight_layout()
    st.pyplot(fig)
