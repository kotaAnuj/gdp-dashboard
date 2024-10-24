
# config.py
import os

GOOGLE_API_KEY = "AIzaSyA5HtRnzGruiia-aKtMMLnBjJ0ovTh11nE"

WIKIPEDIA_USER_AGENT = "WebsiteBuilderBot/1.0"

# Theme configuration
CLAUDE_COLORS = {
    'primary': '#FF6B3D',
    'secondary': '#FF8F6B',
    'background': '#FFFFFF',
    'text': '#1A1A1A',
    'accent': '#FFE4DC'
}

# agents/base_agent.py
from abc import ABC, abstractmethod
import google.generativeai as genai
from typing import Dict, Any, List
import json
import logging

logging.basicConfig(level=logging.INFO)

class BaseAgent(ABC):
    def __init__(self, model: Any, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    @abstractmethod
    def process(self, *args, **kwargs):
        pass

# agents/label_creator_agent.py
class LabelCreatorAgent(BaseAgent):
    def process(self, user_prompt: str) -> Dict:
        self.logger.info("Creating labels from user prompt")
        prompt = f"""
        Analyze and create detailed labels for website requirements:
        {user_prompt}

        Generate a comprehensive JSON structure with:
        1. Page Structure:
           - Layout components
           - Navigation elements
           - Content sections
        2. Design Requirements:
           - Color scheme
           - Typography
           - Spacing and layout rules
        3. Functionality:
           - Interactive elements
           - Forms and inputs
           - Dynamic features
        4. Content Requirements:
           - Text sections
           - Media elements
           - Data requirements
        5. Technical Specifications:
           - Required libraries
           - API integrations
           - Performance requirements
        """
        try:
            response = self.generate_response(prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {str(e)}")
            raise

# agents/task_distributor_agent.py
class TaskDistributorAgent(BaseAgent):
    def process(self, labels: Dict) -> List[Dict]:
        self.logger.info("Distributing tasks based on labels")
        prompt = f"""
        Create a detailed task distribution plan for:
        {json.dumps(labels, indent=2)}

        Generate tasks for:
        1. Frontend Development:
           - Component creation
           - Styling implementation
           - Responsive design
        2. Backend Integration:
           - API endpoints
           - Data processing
           - Security features
        3. Content Generation:
           - Text content
           - Media assets
           - SEO elements
        4. Testing Requirements:
           - Unit tests
           - Integration tests
           - Performance testing

        Return as JSON array with task details, dependencies, and priorities.
        """
        try:
            response = self.generate_response(prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {str(e)}")
            raise

# agents/coding_agent.py
class CodingAgent(BaseAgent):
    def process(self, task: Dict) -> Dict:
        self.logger.info(f"Generating code for task: {task.get('name', 'Unknown')}")
        prompt = f"""
        Generate production-ready code for:
        {json.dumps(task, indent=2)}

        Include:
        1. Complete component code
        2. Styling (CSS/SCSS)
        3. JavaScript functionality
        4. Error handling
        5. Documentation
        6. Performance optimizations

        Follow best practices for:
        - Clean code
        - Accessibility
        - SEO
        - Performance
        """
        try:
            code_response = self.generate_response(prompt)
            return {
                "task": task,
                "code": code_response,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            raise

# agents/design_agent.py
class DesignAgent(BaseAgent):
    def process(self, labels: Dict) -> Dict:
        self.logger.info("Generating design system")
        prompt = f"""
        Create a comprehensive design system based on:
        {json.dumps(labels, indent=2)}

        Include:
        1. Color Palette:
           - Primary colors
           - Secondary colors
           - Accent colors
           - Semantic colors
        2. Typography:
           - Font families
           - Font sizes
           - Line heights
           - Font weights
        3. Spacing System:
           - Grid system
           - Margins
           - Paddings
        4. Component Styles:
           - Buttons
           - Forms
           - Cards
           - Navigation
        5. Animation Guidelines:
           - Transitions
           - Hover states
           - Loading states
        """
        try:
            design_response = self.generate_response(prompt)
            return json.loads(design_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing design system JSON: {str(e)}")
            raise

# agents/upgrading_agent.py
class UpgradingAgent(BaseAgent):
    def process(self, component: Dict) -> Dict:
        self.logger.info(f"Upgrading component: {component.get('task', {}).get('name', 'Unknown')}")
        prompt = f"""
        Optimize and upgrade the following component:
        {json.dumps(component, indent=2)}

        Focus on:
        1. Performance Optimization:
           - Code splitting
           - Lazy loading
           - Resource optimization
        2. Security Improvements:
           - Input validation
           - XSS prevention
           - CSRF protection
        3. Accessibility Enhancements:
           - ARIA labels
           - Keyboard navigation
           - Screen reader support
        4. SEO Optimization:
           - Meta tags
           - Semantic HTML
           - Schema markup
        """
        try:
            upgraded_code = self.generate_response(prompt)
            return {
                "task": component["task"],
                "original_code": component["code"],
                "upgraded_code": upgraded_code,
                "upgrade_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.1"
                }
            }
        except Exception as e:
            self.logger.error(f"Error upgrading component: {str(e)}")
            raise

# agents/verification_agent.py
class VerificationAgent(BaseAgent):
    def process(self, website_data: Dict) -> Dict:
        self.logger.info("Verifying website components and integration")
        prompt = f"""
        Perform comprehensive verification of:
        {json.dumps(website_data, indent=2)}

        Check for:
        1. Code Quality:
           - Syntax validation
           - Best practices
           - Code standards
        2. Integration:
           - Component compatibility
           - API integration
           - State management
        3. Performance:
           - Load time optimization
           - Resource usage
           - Memory leaks
        4. Security:
           - Vulnerability scanning
           - Input validation
           - Authentication checks
        5. Accessibility:
           - WCAG compliance
           - Screen reader compatibility
           - Keyboard navigation
        """
        try:
            verification_response = self.generate_response(prompt)
            return json.loads(verification_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing verification results: {str(e)}")
            raise

# utils/wiki_content_generator.py
class WikiContentGenerator:
    def __init__(self, wikipedia_api_wrapper):
        self.wiki = wikipedia_api_wrapper

    def generate_content(self, topic: str) -> Dict:
        try:
            search_results = self.wiki.search(topic)
            content = []
            for result in search_results[:3]:
                try:
                    page = self.wiki.page(result)
                    content.append({
                        "title": page.title,
                        "summary": page.summary,
                        "url": page.url
                    })
                except Exception as e:
                    logging.error(f"Error fetching wiki page {result}: {str(e)}")
                    continue
            return {"status": "success", "content": content}
        except Exception as e:
            logging.error(f"Error in wiki content generation: {str(e)}")
            return {"status": "error", "message": str(e)}

# website_builder.py
class WebsiteBuilder:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.agents = {
            "label_creator": LabelCreatorAgent(self.model),
            "task_distributor": TaskDistributorAgent(self.model),
            "coding": CodingAgent(self.model),
            "upgrading": UpgradingAgent(self.model),
            "design": DesignAgent(self.model),
            "verification": VerificationAgent(self.model)
        }
        self.wiki_generator = WikiContentGenerator(WikipediaAPIWrapper())
        self.logger = logging.getLogger("WebsiteBuilder")

    def build_website(self, user_prompt: str) -> Dict:
        try:
            # 1. Create labels
            self.logger.info("Step 1: Creating labels")
            labels = self.agents["label_creator"].process(user_prompt)

            # 2. Generate content from Wikipedia if needed
            if "content_requirements" in labels:
                self.logger.info("Generating content from Wikipedia")
                for topic in labels["content_requirements"]:
                    wiki_content = self.wiki_generator.generate_content(topic)
                    labels["wiki_content"] = wiki_content

            # 3. Create design system
            self.logger.info("Step 2: Creating design system")
            design_system = self.agents["design"].process(labels)

            # 4. Distribute tasks
            self.logger.info("Step 3: Distributing tasks")
            tasks = self.agents["task_distributor"].process(labels)

            # 5. Generate components in parallel
            self.logger.info("Step 4: Generating components")
            components = []
            for task in tasks:
                component = self.agents["coding"].process(task)
                components.append(component)

            # 6. Upgrade components
            self.logger.info("Step 5: Upgrading components")
            upgraded_components = []
            for component in components:
                upgraded = self.agents["upgrading"].process(component)
                upgraded_components.append(upgraded)

            # 7. Verify website
            self.logger.info("Step 6: Verifying website")
            verification_results = self.agents["verification"].process({
                "components": upgraded_components,
                "design_system": design_system,
                "labels": labels
            })

            # 8. Generate preview HTML
            preview_html = self.generate_preview_html(upgraded_components, design_system)

            return {
                "status": "success",
                "labels": labels,
                "design_system": design_system,
                "components": upgraded_components,
                "verification": verification_results,
                "preview_html": preview_html
            }
        except Exception as e:
            self.logger.error(f"Error building website: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def generate_preview_html(self, components: List[Dict], design_system: Dict) -> str:
        # [Previous generate_preview_html code remains the same]
        pass

# main.py
def main():
    st.set_page_config(page_title="AI Website Builder", layout="wide")
    
    # Initialize session state
    if 'build_history' not in st.session_state:
        st.session_state.build_history = []
    
    if 'current_build' not in st.session_state:
        st.session_state.current_build = None

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f'''
        <div style="background-color: {CLAUDE_COLORS["accent"]}; padding: 1rem; border-radius: 4px;">
            <h2>Configuration</h2>
        </div>
        ''', unsafe_allow_html=True)
        
        # Build History
        if st.session_state.build_history:
            st.subheader("Build History")
            for idx, build in enumerate(st.session_state.build_history):
                if st.button(f"Build {idx + 1}", key=f"build_{idx}"):
                    st.session_state.current_build = build

    # Main content area
    st.title("AI Website Builder")
    
    # Three-column layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Website Requirements")
        user_prompt = st.text_area("Describe your website", height=200)
        
        if st.button("Generate Website", type="primary"):
            if user_prompt:
                with st.spinner("Building your website..."):
                    builder = WebsiteBuilder()
                    result = builder.build_website(user_prompt)
                    
                    if result["status"] == "success":
                        st.session_state.current_build = result
                        st.session_state.build_history.append(result)
                        st.success("Website generated successfully!")
                    else:
                        st.error(f"Error: {result['message']}")
            else:
                st.error("Please provide website requirements")
    
    # Display current build
    if st.session_state.current_build:
        # Code Column
        with col2:
            st.subheader("Generated Code")
            tabs = st.tabs(["Components", "Design System", "Verification"])
            
            with tabs[0]:
                for idx, component in enumerate(st.session_state.current_build["components"]):
                    with st.expander(f"Component {idx + 1}"):
                        st.code(component["upgraded_code"], language="html")
                        if st.button(f"Download Component {idx + 1}", key=f"download_comp_{idx}"):
                            st.download_button(
                                label=f"Download Component {idx + 1}",
                                data=component["upgraded_code"],
                                file_name=f"component_{idx + 1}.html",
                                mime="text/html"
                            )
            
            with tabs[1]:
                st.json(st.session_state.current_build["design_system"])
            
            with tabs[2]:
                # Continuing from the previous code...

                st.json(st.session_state.current_build["verification"])

        # Preview Column
        with col3:
            st.subheader("Live Preview")
            preview_container = st.container()
            
            # Preview controls
            col3_1, col3_2, col3_3 = st.columns(3)
            with col3_1:
                preview_device = st.selectbox(
                    "Preview Device",
                    ["Desktop", "Tablet", "Mobile"]
                )
            
            with col3_2:
                preview_theme = st.selectbox(
                    "Preview Theme",
                    ["Light", "Dark"]
                )
            
            with col3_3:
                if st.button("Refresh Preview"):
                    st.experimental_rerun()
            
            # Preview window
            preview_html = st.session_state.current_build["preview_html"]
            
            # Adjust preview based on device selection
            preview_width = {
                "Desktop": "100%",
                "Tablet": "768px",
                "Mobile": "375px"
            }[preview_device]
            
            # Apply theme to preview
            if preview_theme == "Dark":
                preview_html = preview_html.replace(
                    'background-color: var(--background-color)',
                    'background-color: #1a1a1a; color: #ffffff'
                )
            
            # Display preview
            components.html(
                f"""
                <div style="width: {preview_width}; margin: 0 auto; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
                    {preview_html}
                </div>
                """,
                height=600,
                scrolling=True
            )
            
            # Export options
            st.subheader("Export Options")
            export_cols = st.columns(2)
            
            with export_cols[0]:
                if st.button("Download Complete Website"):
                    # Create zip file containing all components
                    with ZipFile("website.zip", "w") as zipf:
                        # Add HTML file
                        zipf.writestr("index.html", preview_html)
                        
                        # Add CSS
                        css = st.session_state.current_build["design_system"].get("css", "")
                        zipf.writestr("styles.css", css)
                        
                        # Add JavaScript
                        js = st.session_state.current_build["design_system"].get("javascript", "")
                        zipf.writestr("scripts.js", js)
                        
                        # Add components
                        for idx, component in enumerate(st.session_state.current_build["components"]):
                            zipf.writestr(
                                f"components/component_{idx + 1}.html",
                                component["upgraded_code"]
                            )
                    
                    # Provide download button
                    with open("website.zip", "rb") as f:
                        st.download_button(
                            label="Download ZIP",
                            data=f,
                            file_name="website.zip",
                            mime="application/zip"
                        )
            
            with export_cols[1]:
                if st.button("Export to GitHub"):
                    st.info("GitHub export functionality coming soon!")

# Add utility functions
def create_github_repository(components: List[Dict], design_system: Dict) -> str:
    """Create a GitHub repository with the generated website code"""
    # Implementation for GitHub integration
    pass

def optimize_images(html_content: str) -> str:
    """Optimize images in the HTML content"""
    from bs4 import BeautifulSoup
    import requests
    from PIL import Image
    from io import BytesIO
    
    soup = BeautifulSoup(html_content, 'html.parser')
    images = soup.find_all('img')
    
    for img in images:
        src = img.get('src')
        if src and src.startswith('http'):
            try:
                # Download image
                response = requests.get(src)
                img_data = Image.open(BytesIO(response.content))
                
                # Optimize image
                optimized_data = BytesIO()
                img_data.save(optimized_data, format=img_data.format, optimize=True, quality=85)
                
                # Convert to base64
                import base64
                base64_img = base64.b64encode(optimized_data.getvalue()).decode()
                img['src'] = f"data:image/{img_data.format.lower()};base64,{base64_img}"
            except Exception as e:
                logging.error(f"Error optimizing image {src}: {str(e)}")
    
    return str(soup)

def minify_code(code: str, code_type: str) -> str:
    """Minify HTML, CSS, or JavaScript code"""
    try:
        if code_type == 'html':
            from htmlmin import minify
            return minify(code)
        elif code_type == 'css':
            from rcssmin import cssmin
            return cssmin(code)
        elif code_type == 'js':
            from jsmin import jsmin
            return jsmin(code)
        return code
    except Exception as e:
        logging.error(f"Error minifying {code_type} code: {str(e)}")
        return code

# Add performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def start_operation(self, operation_name: str):
        self.metrics[operation_name] = {
            'start_time': time.time()
        }
    
    def end_operation(self, operation_name: str):
        if operation_name in self.metrics:
            self.metrics[operation_name]['end_time'] = time.time()
            self.metrics[operation_name]['duration'] = (
                self.metrics[operation_name]['end_time'] - 
                self.metrics[operation_name]['start_time']
            )
    
    def get_metrics(self) -> Dict:
        return {
            name: data['duration']
            for name, data in self.metrics.items()
            if 'duration' in data
        }

# Add error handling middleware
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

# Add caching
@st.cache_data(ttl=3600)
def cache_website_build(user_prompt: str) -> Dict:
    """Cache website build results for 1 hour"""
    builder = WebsiteBuilder()
    return builder.build_website(user_prompt)

if __name__ == "__main__":
    try:
        # Initialize performance monitoring
        performance_monitor = PerformanceMonitor()
        performance_monitor.start_operation("main_execution")
        
        # Run the main application
        main()
        
        # End performance monitoring
        performance_monitor.end_operation("main_execution")
        
        # Log performance metrics
        metrics = performance_monitor.get_metrics()
        logging.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")
