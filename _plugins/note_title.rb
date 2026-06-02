# Ensure label always wins over Jekyll's auto-generated collection title.
module Jekyll
  class NoteLabelTitle < Generator
    safe true
    priority :lowest

    def generate(site)
      collection = site.collections["notes"]
      return unless collection

      collection.docs.each do |doc|
        label = doc.data["label"]
        next unless label && !label.to_s.strip.empty?

        doc.data["title"] = label.to_s.strip
      end
    end
  end
end
