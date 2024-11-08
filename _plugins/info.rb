require 'liquid'

class InfoPrompt < Liquid::Block
  def initialize(tag_name, markup, tokens)
    super
  end

  def render(context)
    content = super.strip  # This gets the content between the opening and closing tags

    # Inline CSS
    css = <<~CSS
      <style>
        .info-prompt {
            background-color: #e8f8ed; /* Light green for a soothing effect */
            border-left: 5px solid #3a9f04; /* Darker green for emphasis */
            padding: 15px;
            margin: 20px 0;
            font-family: Cambria, Cochin, Georgia, Times, 'Times New Roman', serif;
            font-size: 1em;
            color: #333;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Optional shadow for depth */
        }
      </style>
    CSS

    <<~HTML
      #{css}
      <p class="info-prompt">#{content}</p>
    HTML
  end
end

Liquid::Template.register_tag('info', InfoPrompt)
