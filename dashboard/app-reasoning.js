/**
 * Apple-Quality Reasoning Panel Integration
 * Integrates the ReasoningPanel into the dashboard
 */

import { ReasoningPanel } from './ReasoningPanel.js';

// Initialize reasoning panel when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  // Create global reasoning panel instance
  window.ReasoningPanel = {
    instances: new Map(),
    
    /**
     * Show reasoning panel for a specific game
     */
    showReasoning(game, prediction, enrichment) {
      const container = document.getElementById('reasoning-panel-container');
      if (!container) {
        this.createContainer();
      }
      
      const container = document.getElementById('reasoning-panel-container');
      if (!container) return;
      
      // Render the reasoning panel
      container.innerHTML = this.renderReasoningPanel(game, prediction, enrichment);
      container.classList.remove('hidden');
      container.classList.add('visible');
      
      // Animate in
      requestAnimationFrame(() => {
        container.classList.add('visible');
      });
      
      // Bind tab events
      this.bindTabEvents(container);
      
      // Store current game
      this.currentGame = game;
      this.currentPrediction = prediction;
      this.currentEnrichment = enrichment;
    },
    
    /**
     * Hide the reasoning panel
     */
    hideReasoning() {
      const container = document.getElementById('reasoning-panel-container');
      if (!container) return;
      
      container.classList.remove('visible');
      setTimeout(() => {
        container.classList.add('hidden');
      }, 300);
    },
    
    /**
     * Render the complete reasoning panel
     */
    render(game, prediction, enrichment) {
      if (!prediction || !prediction.outcomeLabel) {
        return '<div class="reasoning-empty">No prediction available for this game</div>';
      }
      
      const predictedSide = prediction.predictedSide;
      const homeTeam = game.homeTeam;
      const awayTeam = game.awayTeam;
      const predictedTeam = prediction.predictedSide === 'home' ? game.homeTeam : game.awayTeam;
      const otherTeam = predictedTeam === game.homeTeam ? game.awayTeam : game.homeTeam;
      const confidence = prediction.confidence || 0;
      const factors = prediction.factors || [];
      const reasons = prediction.reasons || [];
      const sources = prediction.dataSources || [];
      
      const homeTeamData = game.enrichment?.homeAdvanced || {};
      const awayTeamData = game.enrichment?.awayAdvanced || {};
      
      return `
        <div class="reasoning-panel" data-game-id="${game.eventId}">
          ${this.renderHeader(game, prediction)}
          <div class="reasoning-tabs" role="tablist">
            ${this.tabs.map(tab => `
              <button class="reasoning-tab ${tab.id === this.activeTab ? 'active' : ''}" 
                      role="tab" 
                      aria-selected="${tab.id === this.activeTab}"
                      data-tab="${tab.id}"
                      aria-controls="panel-${tab.id}">
                <span class="tab-icon">${tab.icon}</span>
                <span class="tab-label">${tab.label}</span>
              </button>
            `).join('')}
        </div>
  
        <div class="reasoning-panels">
          ${this.renderFactorsPanel(prediction, game)}
          ${this.renderComparisonPanel(game)}
          ${this.renderHistoryPanel()}
          ${this.renderCounterPanel(game, prediction)}
        </div>
      </div>
    `;
  }
};
EOF
echo "Created ReasoningPanel.js"