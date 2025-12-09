"""
LLM Evaluation Framework - Streamlit UI
Interactive dashboard for benchmarking and comparing LLM performance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from evaluator import LLMEvaluator, ModelPricing
from prompt_optimizer import CostOptimizer, ContextWindowOptimizer
from benchmarks import BenchmarkSuite, TaskCategory, BenchmarkTask

# Page config
st.set_page_config(
    page_title="LLM Evaluation Framework | Gabriel Ordonez",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #0f3460;
        margin-bottom: 1rem;
    }
    .big-metric {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00e5ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .winner-badge {
        background: linear-gradient(135deg, #3d5afe 0%, #00e5ff 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
    if "optimizer" not in st.session_state:
        st.session_state.optimizer = CostOptimizer()
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None
    if "optimization_history" not in st.session_state:
        st.session_state.optimization_history = []


def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")

        st.subheader("Model Selection")
        models = st.multiselect(
            "Select models to compare",
            ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            default=["claude-3-sonnet", "gpt-4-turbo"]
        )

        st.subheader("Benchmark Settings")
        runs_per_model = st.slider("Runs per model", 1, 10, 3)

        categories = st.multiselect(
            "Task categories",
            [c.value for c in TaskCategory],
            default=["reasoning", "coding"]
        )

        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        **LLM Evaluation Framework**

        Compare LLM performance across:
        - 📊 Latency & Throughput
        - 💰 Cost Analysis
        - 🎯 Quality Metrics
        - 🔧 Prompt Optimization

        Built with Python, Streamlit, and Plotly.
        """)

        return models, runs_per_model, [TaskCategory(c) for c in categories]


def render_benchmark_tab(models, runs_per_model, categories):
    """Render the benchmarking tab"""
    st.header("📊 Model Benchmarking")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Run comprehensive benchmarks comparing LLM performance across multiple dimensions.
        Select models and categories in the sidebar, then click **Run Benchmark**.
        """)

    with col2:
        if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
            with st.spinner("Running benchmarks... This may take a few minutes."):
                suite = BenchmarkSuite()
                results = suite.run_full_suite(
                    models=models,
                    runs_per_model=runs_per_model,
                    categories=categories
                )
                st.session_state.benchmark_results = results
                st.success("Benchmark complete!")

    # Display results if available
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results

        # Summary metrics
        st.subheader("Performance Summary")

        comparison = results.get("comparison", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Fastest</div>
                <div class="big-metric">{comparison.get('fastest', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Cheapest</div>
                <div class="big-metric">{comparison.get('cheapest', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Highest Throughput</div>
                <div class="big-metric">{comparison.get('highest_throughput', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Best Quality</div>
                <div class="big-metric">{comparison.get('highest_quality', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        st.subheader("Detailed Comparison")

        # Prepare data for charts
        summaries = results.get("model_summaries", [])

        if summaries:
            df = pd.DataFrame(summaries)

            col1, col2 = st.columns(2)

            with col1:
                # Latency comparison
                fig = px.bar(
                    df,
                    x="model",
                    y=["avg_latency_ms", "p95_latency_ms"],
                    barmode="group",
                    title="Latency Comparison (ms)",
                    color_discrete_sequence=["#3d5afe", "#00e5ff"]
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Throughput comparison
                fig = px.bar(
                    df,
                    x="model",
                    y="tokens_per_second",
                    title="Throughput (tokens/second)",
                    color="model",
                    color_discrete_sequence=["#3d5afe", "#00e5ff", "#ff6b6b"]
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Cost comparison
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    df,
                    values="total_cost_usd",
                    names="model",
                    title="Cost Distribution",
                    color_discrete_sequence=["#3d5afe", "#00e5ff", "#ff6b6b"]
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Quality scores if available
                if "avg_quality_score" in df.columns and df["avg_quality_score"].notna().any():
                    fig = px.bar(
                        df,
                        x="model",
                        y="avg_quality_score",
                        title="Average Quality Score",
                        color="model",
                        color_discrete_sequence=["#3d5afe", "#00e5ff"]
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Task-level results
        st.subheader("Task-Level Results")

        task_results = results.get("task_results", [])
        if task_results:
            task_df = []
            for task in task_results:
                for model, metrics in task["models"].items():
                    task_df.append({
                        "Task": task["task"],
                        "Category": task["category"],
                        "Model": model,
                        "Latency (ms)": metrics["avg_latency_ms"],
                        "Quality": metrics["avg_quality"],
                        "Cost ($)": metrics["total_cost"]
                    })

            st.dataframe(
                pd.DataFrame(task_df),
                use_container_width=True,
                hide_index=True
            )


def render_optimizer_tab():
    """Render the prompt optimization tab"""
    st.header("🔧 Prompt Optimizer")

    st.markdown("""
    Optimize your prompts to reduce token usage and costs.
    The optimizer applies multiple techniques to compress prompts while preserving meaning.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Prompt")
        input_prompt = st.text_area(
            "Enter your prompt",
            height=200,
            placeholder="Paste your prompt here to optimize...",
            value="""Please could you kindly help me understand, in order to better grasp the concept,
what machine learning is? I would like you to explain it in simple terms.
Due to the fact that I am a beginner, please use analogies.
It is important to note that I have no prior experience with programming."""
        )

        model = st.selectbox(
            "Target model",
            ["claude-3-sonnet", "claude-3-opus", "gpt-4-turbo", "gpt-4"]
        )

        aggressive = st.checkbox("Aggressive optimization", help="Apply more aggressive compression techniques")

        if st.button("⚡ Optimize Prompt", type="primary"):
            if input_prompt:
                optimizer = ContextWindowOptimizer()
                result = optimizer.optimize(input_prompt, model, aggressive)

                st.session_state.optimization_history.append({
                    "original": result.original_tokens,
                    "optimized": result.optimized_tokens,
                    "saved": result.tokens_saved,
                    "percent": result.reduction_percent
                })

                with col2:
                    st.subheader("Optimized Prompt")
                    st.text_area("Result", result.optimized_prompt, height=200, disabled=True)

                    # Metrics
                    st.markdown("### Optimization Results")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Original Tokens", result.original_tokens)
                    m2.metric("Optimized Tokens", result.optimized_tokens, f"-{result.tokens_saved}")
                    m3.metric("Reduction", f"{result.reduction_percent}%")

                    if result.techniques_applied:
                        st.markdown("**Techniques Applied:**")
                        for tech in result.techniques_applied:
                            st.markdown(f"- {tech.replace('_', ' ').title()}")

    # Show optimization history
    if st.session_state.optimization_history:
        st.markdown("---")
        st.subheader("📈 Optimization History")

        history_df = pd.DataFrame(st.session_state.optimization_history)
        history_df["run"] = range(1, len(history_df) + 1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Original",
            x=history_df["run"],
            y=history_df["original"],
            marker_color="#ff6b6b"
        ))
        fig.add_trace(go.Bar(
            name="Optimized",
            x=history_df["run"],
            y=history_df["optimized"],
            marker_color="#00e5ff"
        ))

        fig.update_layout(
            barmode="group",
            title="Token Reduction Over Time",
            xaxis_title="Optimization Run",
            yaxis_title="Tokens",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        total_saved = sum(h["saved"] for h in st.session_state.optimization_history)
        avg_reduction = sum(h["percent"] for h in st.session_state.optimization_history) / len(st.session_state.optimization_history)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tokens Saved", total_saved)
        col2.metric("Average Reduction", f"{avg_reduction:.1f}%")
        col3.metric("Estimated Cost Saved", f"${total_saved * 0.00001:.4f}")


def render_pricing_tab():
    """Render the pricing calculator tab"""
    st.header("💰 Cost Calculator")

    st.markdown("""
    Calculate and compare costs across different models based on expected usage.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Usage Estimation")

        daily_requests = st.number_input("Daily requests", min_value=0, value=1000)
        avg_input_tokens = st.number_input("Avg input tokens per request", min_value=0, value=500)
        avg_output_tokens = st.number_input("Avg output tokens per request", min_value=0, value=200)

        models_to_compare = st.multiselect(
            "Models to compare",
            list(ModelPricing.PRICING.keys()),
            default=["claude-3-sonnet", "gpt-4-turbo", "gpt-3.5-turbo"]
        )

    with col2:
        st.subheader("Cost Projection")

        if models_to_compare:
            cost_data = []

            for model in models_to_compare:
                daily_cost = ModelPricing.calculate_cost(
                    model,
                    daily_requests * avg_input_tokens,
                    daily_requests * avg_output_tokens
                )
                monthly_cost = daily_cost * 30
                yearly_cost = daily_cost * 365

                cost_data.append({
                    "Model": model,
                    "Daily": f"${daily_cost:.2f}",
                    "Monthly": f"${monthly_cost:.2f}",
                    "Yearly": f"${yearly_cost:.2f}"
                })

            st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

            # Chart
            chart_data = []
            for model in models_to_compare:
                daily_cost = ModelPricing.calculate_cost(
                    model,
                    daily_requests * avg_input_tokens,
                    daily_requests * avg_output_tokens
                )
                chart_data.append({"model": model, "daily_cost": daily_cost})

            fig = px.bar(
                pd.DataFrame(chart_data),
                x="model",
                y="daily_cost",
                title="Daily Cost Comparison",
                color="model",
                color_discrete_sequence=["#3d5afe", "#00e5ff", "#ff6b6b", "#ffeb3b"]
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
                showlegend=False,
                yaxis_title="Cost (USD)"
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point"""
    init_session_state()

    st.title("📊 LLM Evaluation Framework")
    st.markdown("*Benchmark, compare, and optimize LLM performance*")

    models, runs_per_model, categories = render_sidebar()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Benchmarks", "🔧 Optimizer", "💰 Pricing"])

    with tab1:
        render_benchmark_tab(models, runs_per_model, categories)

    with tab2:
        render_optimizer_tab()

    with tab3:
        render_pricing_tab()


if __name__ == "__main__":
    main()
