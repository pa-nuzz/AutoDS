"""Interactive checklist for guiding users through analysis steps.

Provides a visual task list that updates based on completed actions.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class ChecklistItem:
    """Single item in the checklist."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    category: str = "general"
    depends_on: List[str] = field(default_factory=list)
    action_text: str = "Start"
    help_text: str = ""
    estimated_time: str = "2-5 min"
    icon: str = "⭕"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'category': self.category,
            'depends_on': self.depends_on,
            'action_text': self.action_text,
            'help_text': self.help_text,
            'estimated_time': self.estimated_time,
            'icon': self.icon
        }


class AnalysisChecklist:
    """Interactive checklist for data analysis workflow.
    
    Guides users through:
    1. Data Upload
    2. Data Exploration
    3. Preprocessing
    4. Model Training
    5. Report Generation
    """
    
    def __init__(self):
        self.items: Dict[str, ChecklistItem] = {}
        self._init_default_checklist()
    
    def _init_default_checklist(self):
        """Initialize the default analysis workflow checklist."""
        default_items = [
            ChecklistItem(
                id="upload",
                title="Upload Your Data",
                description="Import your dataset from a file or URL. We support CSV, Excel, JSON, and more.",
                category="data",
                action_text="Upload Data",
                help_text="Go to the Upload tab and drag-drop your file, or paste a URL from Kaggle/GitHub.",
                estimated_time="1-2 min",
                icon="📤"
            ),
            ChecklistItem(
                id="explore",
                title="Explore & Profile",
                description="Get an overview of your data with automatic profiling and visualizations.",
                category="analysis",
                depends_on=["upload"],
                action_text="Explore Data",
                help_text="Visit the Explore tab to see data distributions, correlations, and quality metrics.",
                estimated_time="3-5 min",
                icon="🔍"
            ),
            ChecklistItem(
                id="select_target",
                title="Select Target Variable",
                description="Choose which column you want to predict (for supervised learning).",
                category="analysis",
                depends_on=["explore"],
                action_text="Set Target",
                help_text="In Explore or Preprocess tab, select your target column from the dropdown.",
                estimated_time="1 min",
                icon="🎯"
            ),
            ChecklistItem(
                id="preprocess",
                title="Preprocess Data",
                description="Clean and prepare your data using Auto mode or step-by-step DIY guidance.",
                category="processing",
                depends_on=["explore"],
                action_text="Preprocess",
                help_text="In the Preprocess tab, choose 'Do It For Me' for automatic cleaning, or 'DIY' for guided steps.",
                estimated_time="3-5 min",
                icon="🔧"
            ),
            ChecklistItem(
                id="get_recommendations",
                title="Get Model Recommendations",
                description="See which machine learning models are best suited for your data and task.",
                category="modeling",
                depends_on=["preprocess", "select_target"],
                action_text="Get Recommendations",
                help_text="Visit the Models tab and click 'Get Recommendations' to see suggested algorithms.",
                estimated_time="1-2 min",
                icon="💡"
            ),
            ChecklistItem(
                id="train_models",
                title="Train Baseline Models",
                description="Automatically train multiple models and compare their performance.",
                category="modeling",
                depends_on=["get_recommendations"],
                action_text="Train Models",
                help_text="In the Models tab, click 'Train Models' to start baseline training. Takes a few minutes.",
                estimated_time="5-10 min",
                icon="🤖"
            ),
            ChecklistItem(
                id="generate_report",
                title="Generate Full Report",
                description="Create a comprehensive report with all analysis, charts, and insights.",
                category="reporting",
                depends_on=["explore"],
                action_text="Generate Report",
                help_text="In the Reports tab, choose your export formats and click 'Generate Report'.",
                estimated_time="2-3 min",
                icon="📊"
            ),
            ChecklistItem(
                id="export_results",
                title="Export Results",
                description="Download your cleaned data, model, and reports in various formats.",
                category="reporting",
                depends_on=["generate_report"],
                action_text="Export",
                help_text="In Reports tab or after Auto Mode, download your files using the export buttons.",
                estimated_time="1 min",
                icon="💾"
            ),
            ChecklistItem(
                id="auto_mode",
                title="Try Auto Mode (Optional)",
                description="Run the complete pipeline in one click - upload to report automatically.",
                category="workflow",
                depends_on=["upload"],
                action_text="Run Auto Mode",
                help_text="Visit the Auto Mode tab, select your target, and click 'Start Complete Analysis'.",
                estimated_time="10-15 min",
                icon="⚡"
            ),
        ]
        
        for item in default_items:
            self.items[item.id] = item
    
    def update_status(self, item_id: str, status: TaskStatus):
        """Update the status of a checklist item."""
        if item_id in self.items:
            self.items[item_id].status = status
    
    def mark_completed(self, item_id: str):
        """Mark an item as completed."""
        self.update_status(item_id, TaskStatus.COMPLETED)
    
    def mark_in_progress(self, item_id: str):
        """Mark an item as in progress."""
        self.update_status(item_id, TaskStatus.IN_PROGRESS)
    
    def get_available_items(self) -> List[ChecklistItem]:
        """Get items that are ready to be worked on.
        
        Items are available if:
        - Their dependencies are completed
        - They are pending or in progress
        """
        available = []
        for item in self.items.values():
            # Check if dependencies are met
            deps_met = all(
                self.items[dep].status == TaskStatus.COMPLETED
                for dep in item.depends_on
                if dep in self.items
            )
            
            if deps_met and item.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                available.append(item)
        
        return available
    
    def get_completed_items(self) -> List[ChecklistItem]:
        """Get all completed items."""
        return [item for item in self.items.values() 
                if item.status == TaskStatus.COMPLETED]
    
    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics."""
        total = len(self.items)
        completed = sum(1 for item in self.items.values() 
                       if item.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for item in self.items.values() 
                         if item.status == TaskStatus.IN_PROGRESS)
        
        return {
            'total': total,
            'completed': completed,
            'in_progress': in_progress,
            'pending': total - completed - in_progress,
            'percentage': (completed / total * 100) if total > 0 else 0,
            'is_complete': completed == total
        }
    
    def get_next_recommended(self) -> Optional[ChecklistItem]:
        """Get the next recommended item to work on."""
        available = self.get_available_items()
        if not available:
            return None
        
        # Prefer in-progress items
        in_progress = [i for i in available if i.status == TaskStatus.IN_PROGRESS]
        if in_progress:
            return in_progress[0]
        
        # Then prefer items with fewer dependencies (earlier in workflow)
        return min(available, key=lambda i: len(i.depends_on))
    
    def to_streamlit_html(self) -> str:
        """Generate HTML for Streamlit display."""
        progress = self.get_progress()
        
        html = f"""
        <div style="
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 1.25rem; font-weight: 600; color: #1e293b;">
                    📋 Analysis Checklist
                </div>
                <div style="
                    background: {'#d1fae5' if progress['percentage'] == 100 else '#dbeafe'};
                    color: {'#059669' if progress['percentage'] == 100 else '#2563eb'};
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                ">
                    {progress['completed']}/{progress['total']} Complete
                </div>
            </div>
            
            <div style="
                background: #e2e8f0;
                height: 8px;
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 1.5rem;
            ">
                <div style="
                    background: linear-gradient(90deg, #3b82f6, #60a5fa);
                    height: 100%;
                    width: {progress['percentage']}%;
                    border-radius: 4px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        """
        
        # Add items by category
        categories = {
            'data': '📊 Data',
            'analysis': '🔍 Analysis', 
            'processing': '⚙️ Processing',
            'modeling': '🤖 Modeling',
            'reporting': '📄 Reporting',
            'workflow': '⚡ Workflow'
        }
        
        for cat_id, cat_name in categories.items():
            cat_items = [i for i in self.items.values() if i.category == cat_id]
            if cat_items:
                html += f'''
                <div style="margin-bottom: 1rem;">
                    <div style="
                        font-size: 0.875rem;
                        font-weight: 600;
                        color: #64748b;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 0.5rem;
                    ">
                        {cat_name}
                    </div>
                '''
                
                for item in cat_items:
                    status_styles = {
                        TaskStatus.PENDING: ('⭕', '#94a3b8', 'opacity: 0.7;'),
                        TaskStatus.IN_PROGRESS: ('🔄', '#3b82f6', 'font-weight: 500;'),
                        TaskStatus.COMPLETED: ('✅', '#10b981', ''),
                        TaskStatus.SKIPPED: ('⏭️', '#f59e0b', 'opacity: 0.7;')
                    }
                    icon, color, extra_style = status_styles.get(item.status, ('⭕', '#94a3b8', ''))
                    
                    html += f'''
                    <div style="
                        display: flex;
                        align-items: flex-start;
                        gap: 0.75rem;
                        padding: 0.75rem;
                        border-radius: 8px;
                        margin-bottom: 0.5rem;
                        background: {'#eff6ff' if item.status == TaskStatus.IN_PROGRESS else 'white'};
                        border: 1px solid {'#bfdbfe' if item.status == TaskStatus.IN_PROGRESS else '#f1f5f9'};
                        {extra_style}
                    ">
                        <div style="font-size: 1.25rem;">{item.icon}</div>
                        <div style="flex: 1;">
                            <div style="
                                font-weight: 500;
                                color: {color};
                                margin-bottom: 0.25rem;
                            ">
                                {icon} {item.title}
                            </div>
                            <div style="font-size: 0.875rem; color: #64748b;">
                                {item.description}
                            </div>
                        </div>
                    </div>
                    '''
                
                html += '</div>'
        
        html += '</div>'
        return html
    
    def save(self, path: str):
        """Save checklist state to file."""
        data = {
            'items': {k: v.to_dict() for k, v in self.items.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load checklist state from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for item_id, item_data in data.get('items', {}).items():
            if item_id in self.items:
                self.items[item_id].status = TaskStatus(item_data.get('status', 'pending'))


# Global checklist instance
checklist = AnalysisChecklist()


def get_checklist() -> AnalysisChecklist:
    """Get the global checklist instance."""
    return checklist
