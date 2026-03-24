// mixer.js — additional utilities loaded from static/js/
// The main application logic lives inline in templates/index.html.
// This file can be used for future extensions or overrides.

(function() {
  'use strict';

  // Expose a small helper for external integrations
  window.DJMixerUtils = {
    formatBpm: function(bpm) {
      if (!bpm) return '000.0';
      return Number(bpm).toFixed(1).padStart(5, '0');
    },
    formatTime: function(sec) {
      sec = Math.round(sec);
      var m = Math.floor(sec / 60);
      var s = sec % 60;
      return m + ':' + String(s).padStart(2, '0');
    },
    bpmCompatibility: function(bpmA, bpmB) {
      // Returns 0-100 compatibility score between two BPMs
      if (!bpmA || !bpmB) return 0;
      var ratio = bpmA > bpmB ? bpmA / bpmB : bpmB / bpmA;
      // harmonics: 1x, 2x, 0.5x
      var distances = [Math.abs(ratio - 1), Math.abs(ratio - 2), Math.abs(ratio - 0.5)];
      var minDist = Math.min.apply(null, distances);
      return Math.max(0, Math.round((1 - minDist / 0.5) * 100));
    },
  };
})();
