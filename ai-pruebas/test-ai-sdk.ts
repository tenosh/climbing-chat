import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import "dotenv/config";

async function testGenerateText() {
  const article = `
    Últimas Noticias del Estado de Sinaloa, México
    Violencia y Protestas en Sinaloa
    El estado de Sinaloa ha comenzado el año 2025 sumido en una ola de violencia intensa. En la primera semana de enero, se reportaron 45 muertes debido a enfrentamientos entre las facciones del crimen organizado Los Chapitos y Los Mayitos, que han estado en conflicto desde septiembre de 20242. La violencia ha afectado principalmente a Culiacán, Mazatlán, Escuinapa, Rosario y Mocorito2.
    Las protestas han sido una respuesta a la inseguridad. Miles de ciudadanos han salido a las calles en Culiacán para exigir el cese de la violencia y criticar al gobernador Rubén Rocha por su gestión35. Las manifestaciones han sido motivadas por incidentes como el asesinato de dos niños junto a su padre, lo que ha colmado la paciencia de la población3.
    Respuesta del Gobierno
    El gobierno federal ha anunciado un refuerzo en Sinaloa para investigar los asesinatos y mejorar la seguridad3. Sin embargo, las declaraciones del gobernador Rocha minimizando la situación han generado indignación entre los ciudadanos5. A pesar del respaldo de figuras políticas como Andrés Manuel López Obrador y Claudia Sheinbaum, la presión sobre Rocha para que dimita sigue creciendo5.
    Operativos y Hallazgos
    Las autoridades han intensificado los operativos en Sinaloa para controlar la violencia. Estos incluyen la requisa de drogas y el despliegue de fuerzas de seguridad para proteger a los ciudadanos4. Además, se han encontrado cuerpos mutilados y se han reportado quema de autos y bloqueos de vías6.
    La situación en Sinaloa sigue siendo crítica, con una población que demanda medidas efectivas para detener la violencia y garantizar su seguridad.
    `;

  try {
    const { text } = await generateText({
      model: openai("gpt-4o-mini"),
      system:
        "You are a sky expert. " +
        "You write simple, short, clear, and concise content.",
      prompt: `Why is the sky red on sunset and not blue?`,
    });
    console.log("Generated Text:", text);
  } catch (error) {
    console.error("Error generating text:", error);
  }
}

testGenerateText();
