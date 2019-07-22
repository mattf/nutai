import os

import connexion

app = connexion.FlaskApp(__name__)
app.add_api('nut.yaml')
app.run(port=os.getenv('PORT'))
