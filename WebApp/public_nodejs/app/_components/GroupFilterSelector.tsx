import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@bluevoid-test/ui/select";
import { groups } from "../_utils/group";

const groupNames = Object.keys(groups).map((value) => ({
  id: value,
  value,
}));

interface GroupFilterSelectorProps {
  onValueChange: (value: string) => void;
  value: string;
  disabled?: boolean;
}
function GroupFilterSelector(props: GroupFilterSelectorProps) {
  const { onValueChange, value, disabled = false } = props;
  return (
    <Select disabled={disabled} onValueChange={onValueChange} value={value}>
      <SelectTrigger className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {groupNames.map((item) => (
          <SelectItem key={item.id} value={item.value}>
            Grupy sektora {item.value}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export default GroupFilterSelector;
