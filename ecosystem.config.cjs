// ecosystem.config.cjs
module.exports = {
  apps: [
    {
      name: 'research_ai',
      cwd: '/home/user/webapp/research_ai',
      script: '/home/user/webapp/research_ai/.venv/bin/streamlit',
      args: 'run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.enableWebsocketCompression false --browser.gatherUsageStats false',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1',
        PORT: 8501
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    },
    {
      name: 'research_ai_preview_3000',
      cwd: '/home/user/webapp/research_ai',
      script: '/home/user/webapp/research_ai/.venv/bin/streamlit',
      args: 'run app.py --server.port 3000 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.enableWebsocketCompression false --browser.gatherUsageStats false',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1',
        PORT: 3000
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    }
  ]
}
