{{ fullname }}
{{ underline }}


.. contents:: Contents
   :local:


.. automodule:: {{fullname}}

   {% block functions %}
   {% if functions %}

   Functions
   =========

   {% for item in functions %}

   {{item}}
   {{ "-" * (item | length) }}
   .. autofunction:: {{ item }}

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

   Classes
   =======
   {% for item in classes %}

   {{item}}
   {{ "-" * (item | length) }}

   .. autoclass:: {{ item }}
     :show-inheritance:
     :special-members: __init__
     :members:
     :undoc-members:

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}

   Exceptions
   ==========

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
