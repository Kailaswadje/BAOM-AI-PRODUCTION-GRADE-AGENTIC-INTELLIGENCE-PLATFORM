from setuptools import setup, find_packages
 
setup(
    name="agentic-intelligence-platform",
    version="0.1.0",
    description="Production-grade agentic intelligence platform: "
                "GNN + Multi-Agent + SLM for domain-agnostic graph reasoning",
    author="[Your Name]",
    packages=find_packages(),
    install_requires=[
        "neo4j>=5.0",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "langchain>=0.2",
        "langchain-ollama>=0.1",
        "langchain-community>=0.2",
        "langgraph>=0.1",
        "crewai>=0.1",
        "transformers>=4.40",
        "streamlit>=1.30",
        "networkx>=3.0",
        "scipy>=1.10",
        "numpy>=1.24",
        "requests>=2.28",
    ],
    python_requires=">=3.11",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)
