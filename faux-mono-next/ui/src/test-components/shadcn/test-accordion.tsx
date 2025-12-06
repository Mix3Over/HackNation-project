import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from "@bluevoid/ui/accordion";
import { Card } from "@bluevoid/ui/card";
import { lorem } from "../lorem";

function TestAccordion() {
	return (
		<Card className="w-full p-2">
			<Accordion type="multiple" className="w-full">
				<AccordionItem value="item-1">
					<AccordionTrigger>Test</AccordionTrigger>
					<AccordionContent>{lorem}</AccordionContent>
				</AccordionItem>
				<AccordionItem value="item-2">
					<AccordionTrigger>Test2</AccordionTrigger>
					<AccordionContent>{lorem}</AccordionContent>
				</AccordionItem>
			</Accordion>
		</Card>
	);
}

export default TestAccordion;
