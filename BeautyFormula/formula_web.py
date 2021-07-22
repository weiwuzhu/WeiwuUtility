from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

import sys
sys.path.append(r"..")
from EssenceFormula import run, beautify_result

app = Flask(__name__)

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

# Flask-Bootstrap requires this line
Bootstrap(app)

# with Flask-WTF, each web form is represented by a class
# "NameForm" can change; "(FlaskForm)" cannot
# see the route for "/" and "index.html" to see how this is used
class EffectForm(FlaskForm):
    field0 = SelectField(u'头香剂', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field1 = SelectField(u'头香新鲜天然', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field2 = SelectField(u'头香透发扩散', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field3 = SelectField(u'主香剂', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field4 = SelectField(u'主香层次丰富', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field5 = SelectField(u'底香定香剂', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field6 = SelectField(u'底香持久留香', choices = [i+1 for i in range(5)], default=1, validators = [DataRequired()])
    field7 = SelectField(u'不含26过敏原', choices = [i for i in range(2)], default=0, validators = [DataRequired()])
    submit = SubmitField('Submit')

class TypeForm(FlaskForm):
    field0 = SelectField(u'品类', choices = [i for i in range(9)], default=0, validators = [DataRequired()])
    field1 = SelectField(u'香型', choices = [i for i in range(12)], default=0, validators = [DataRequired()])
    field2 = SelectField(u'花香种类', choices = [i for i in range(2)], default=0, validators = [DataRequired()])
    field3 = SelectField(u'单花种类', choices = [i for i in range(7)], default=0, validators = [DataRequired()])
    field4 = SelectField(u'香韵', choices = [i for i in range(4)], default=0, validators = [DataRequired()])
    field5 = SelectField(u'有无过敏原料', choices = [i for i in range(2)], default=1, validators = [DataRequired()])
    submit = SubmitField('Submit')

# all Flask routes below

@app.route('/', methods=['GET', 'POST'])
def type():
    # you must tell the variable 'form' what you named the class, above
    # 'form' is the variable name used in this template: index.html
    form = TypeForm()
    type_dict = {
        (0,0,0,0,0,1) : '桂花',
    }
    message = ""
    if form.validate_on_submit():
        field0_value = form.field0.data
        field1_value = form.field1.data
        field2_value = form.field2.data
        field3_value = form.field3.data
        field4_value = form.field4.data
        field5_value = form.field5.data
        effect = (int(field0_value),int(field1_value),int(field2_value),int(field3_value),int(field4_value),int(field5_value))
        if effect in type_dict:
            return redirect( url_for('index', type_name=type_dict[effect]) )
        else:
            message = "当前选择的种类暂时不支持"
    return render_template('type.html', form=form, message=message)

@app.route('/index/<type_name>', methods=['GET', 'POST'])
def index(type_name):
    # you must tell the variable 'form' what you named the class, above
    # 'form' is the variable name used in this template: index.html
    form = EffectForm()
    message = ""
    if form.validate_on_submit():
        field0_value = form.field0.data
        field1_value = form.field1.data
        field2_value = form.field2.data
        field3_value = form.field3.data
        field4_value = form.field4.data
        field5_value = form.field5.data
        field6_value = form.field6.data
        field7_value = form.field7.data
        if field0_value:
            effect = []
            effect.append(field0_value)
            effect.append(field1_value)
            effect.append(field2_value)
            effect.append(field3_value)
            effect.append(field4_value)
            effect.append(field5_value)
            effect.append(field6_value)
            effect.append(field7_value)
            return redirect( url_for('formula', effects=','.join(effect)) )
        else:
            message = "Effect is not valid!"
    return render_template('index.html', form=form, message=message, type_name=type_name)

@app.route('/formula/<effects>')
def formula(effects):
    print(effects)
    effect = [int(x) for x in effects.split(',')]
    formula = run(effect)
    effect_str = beautify_result([], 0, effect)
    effect_sum = sum(effect)
    if not formula:
        # redirect the browser to the error template
        return render_template('404.html'), 404
    else:
        # pass all the data to formula page
        return render_template('formula.html', formula=formula, effect_str=effect_str, effect_sum=effect_sum)
			
@app.route('/actor/<effect>')
def actor(effect):
    # run function to get actor data based on the id in the path
    id, name, photo = get_actor(ACTORS, id)
    if name == "Unknown":
        # redirect the browser to the error template
        return render_template('404.html'), 404
    else:
        # pass all the data for the selected actor to the template
        return render_template('actor.html', id=id, name=name, photo=photo)

# 2 routes to handle errors - they have templates too

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# keep this as is
if __name__ == '__main__':
    app.run(debug=True)
