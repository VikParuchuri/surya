<style>
    .katex-display-container {
        display: inline-block;
        max-width: 100%;
        overflow-x: auto;
        max-height: 100%;
    }

    .katex-inline-container {
        display: inline-block;
        max-width: 100%;
        overflow-x: auto;
        max-height: 100%;
    }
</style>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js" onload="setTimeout(function() {renderMath()})" async></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
<script>
    function htmlUnescape(escapedText) {
      const htmlEntities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' '
      };

      return escapedText.replace(/&amp;|&lt;|&gt;|&quot;|&#39;|&nbsp;/g, match => htmlEntities[match]);
    }

    const renderMath = (function() {
    try {
       const mathElements = document.querySelectorAll('math');

        mathElements.forEach(function(element) {
          let mathContent = element.innerHTML.trim();
          mathContent = htmlUnescape(mathContent);
          const isDisplay = element.getAttribute('display') === 'block';

          const container = document.createElement('span');
          container.className = isDisplay ? 'katex-display-container' : 'katex-inline-container';
          element.parentNode.insertBefore(container, element);

          try {
            katex.render(mathContent, container, {
              displayMode: isDisplay,
              throwOnError: false
            });

          } catch (err) {
            console.error('KaTeX rendering error:', err);
            container.textContent = mathContent; // Fallback to raw text
          }

          element.parentNode.removeChild(element);
        });

        console.log('Math rendering complete with', mathElements.length, 'expressions');
      } catch (err) {
        console.error('Error in renderMath function:', err);
      }
    });
</script>