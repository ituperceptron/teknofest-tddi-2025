// Sidebar functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    const sidebarToggle = null; // disabled
    const menuToggle = null; // disabled
    
    // Check if sidebar exists (only on authenticated pages)
    if (sidebar && mainContent) {
        // Sidebar always open: no-op functions
        function toggleSidebar() {}
        function toggleMobileMenu() {}
        
        // Event listeners
        // disabled: no event listeners for toggling
        
        // Always open: ignore any saved collapsed state
        sidebar.classList.remove('collapsed');
        mainContent.classList.remove('expanded');
        
        // Close mobile menu when clicking outside
        // disabled mobile auto-close behavior
        
        // Handle window resize
        // disabled resize behavior for mobile menu
    }
    
    // Set active navigation link based on current page
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.remove();
            }, 300);
        }, 5000);
    });
    
    // Form validation
    const forms = document.querySelectorAll('form:not(.classification-form):not(.text-summary-form)');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.style.borderColor = '#dc3545';
                    
                    // Remove error styling after user starts typing
                    field.addEventListener('input', function() {
                        this.style.borderColor = '#e1e5e9';
                    });
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Lütfen tüm gerekli alanları doldurun.');
                return;
            }
            
            // Allow form to submit normally for registration and login forms
            if (form.action.includes('/register') || form.action.includes('/login')) {
                return; // Don't prevent default, let the form submit
            }
        });
    });
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Card hover effects removed for better UX
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // disabled keyboard toggle shortcut
        
        // disabled escape behavior for mobile menu
    });
    
    // Add tooltips for better UX
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.getAttribute('data-tooltip');
            tooltip.style.cssText = `
                position: absolute;
                background: #333;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 1000;
                pointer-events: none;
                white-space: nowrap;
            `;
            
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
            
            this.tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', function() {
            if (this.tooltip) {
                this.tooltip.remove();
                this.tooltip = null;
            }
        });
    });
});

// Helper function to manage button loading state
function setButtonLoading(button, isLoading, loadingText = 'İşleniyor...') {
    if (isLoading) {
        button.disabled = true;
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${loadingText}`;
    } else {
        button.disabled = false;
        if (button.dataset.originalText) {
            button.innerHTML = button.dataset.originalText;
        }
    }
} 